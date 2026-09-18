import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields, replace
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        # Load a checkpoint directly on resume; initialize a network only for fresh training.
        if config.experiment.policy.path is not None:
            agent_conf, agent_state = PPOJax.load_agent(config.experiment.policy.path)
            # Check if the loaded agent state has valid parameters
            param_leaves = jax.tree.leaves(agent_state.train_state.params)
            if not param_leaves or any(leaf.size == 0 for leaf in param_leaves):
                raise ValueError("The pretrained checkpoint contains empty policy parameters")
            parameter_norm = jnp.sqrt(
                sum(jnp.sum(jnp.square(leaf)) for leaf in param_leaves)
            )
            checkpoint_step = int(agent_state.train_state.step)
            print(
                f"Loaded pretrained checkpoint: {config.experiment.policy.path} "
                f"(optimizer step={checkpoint_step}, parameter norm={float(parameter_norm):.6f})"
            )
            if checkpoint_step <= 0:
                raise ValueError("The checkpoint has no completed optimizer steps")
        else:
            # No previous checkppoint, initialize a new agent configuration and state.
            agent_conf = PPOJax.init_agent_conf(env, config)
            agent_state = None

        # If there is a pretrained agent state, we need to update the configuration to match the current experiment settings.
        if agent_state is not None:
            # Use the complete current YAML configuration while retaining the
            # checkpoint network and parameters.
            with open_dict(config.experiment):
                config.experiment.num_updates = (
                    config.experiment.total_timesteps
                    // config.experiment.num_steps
                    // config.experiment.num_envs
                )
                config.experiment.minibatch_size = (
                    config.experiment.num_envs
                    * config.experiment.num_steps
                    // config.experiment.num_minibatches
                )
                if config.experiment.validation.num > 0:
                    config.experiment.validation_interval = max(
                        1,
                        config.experiment.num_updates
                        // config.experiment.validation.num,
                    )
                    config.experiment.validation.num = int(
                        config.experiment.num_updates
                        // config.experiment.validation_interval
                    )
                else:
                    config.experiment.validation_interval = config.experiment.num_updates

            agent_conf = replace(
                agent_conf,
                config=config,
                tx=PPOJax._get_optimizer(config),
            )

        print(f"Agent configuration: {agent_conf.config.experiment.env_params.init_state_type}")

        # build training function (by resuming training or restarting from scratch)
        train_fn = (PPOJax.build_resume_train_fn(env, agent_conf)
                    if agent_state is not None
                    else PPOJax.build_train_fn(env, agent_conf))

        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        # If there are multiple seeds, we need to broadcast the agent state to match the number of seeds.
        if agent_state is not None and config.experiment.n_seeds > 1:
            agent_state = jax.tree.map(
                lambda value: jnp.broadcast_to(value, (config.experiment.n_seeds,) + value.shape),
                agent_state,
            )

        # Train the agent
        out = train_fn(_rng, agent_state) if agent_state is not None else train_fn(_rng)

        # save agent state
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        runtime_config = agent_conf.config
        # get the metrics and log them
        if not runtime_config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i+1) % runtime_config.experiment.validation_interval == 0 and runtime_config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

                    # log all measures
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # # metric for used for wandb sweep (optional)
                    # site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    # site_rrotvec = validation_metrics.euclidean_distance.site_rrotvec[i]
                    # site_rvel = validation_metrics.euclidean_distance.site_rvel[i]
                    # run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                    #         step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                           train_state_seed=0)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
