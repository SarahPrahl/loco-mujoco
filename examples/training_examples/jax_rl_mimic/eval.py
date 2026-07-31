import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--headless', action='store_true',
                    help='Render headlessly without a display. Uses OSMesa software rendering '
                         'by default; override with the MUJOCO_GL environment variable.')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
if args.headless:
    # OSMesa software rendering works without a display (no X11/EGL needed)
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    config.experiment.env_params["headless"] = True
else:
    config.experiment.env_params["headless"] = False
# NOTE: do not set goal_type here - the observation space of the saved agent is
# fixed at training time. Adding goals (GoalTrajMimic*) changes the observation
# dimension (e.g. +453 dims) and breaks the loaded policy.
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)

print(f"Video saved at: {env.video_file_path}")
