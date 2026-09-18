from typing import Any, Union, List, Tuple
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported
from loco_mujoco.core.initial_state_handler.base import InitialStateHandler


class DefaultInitialStateHandler(InitialStateHandler):

    """
    Basic initial state handler setting the initial joint positions and velocities. By default,
    the initial joint positions and velocities are set to None, which means that the initial state
    is not modified.

    """

    def __init__(self, env: Any, qpos_init=None, qvel_init=None):
        """
        Initialize the DefaultInitialStateHandler class.

        Args:
            env (Any): The environment instance.
            qpos_init (Union[None, List[float]]): Initial joint positions.
            qvel_init (Union[None, List[float]]): Initial joint velocities.
        """

        self.qpos_init = np.array(qpos_init) if qpos_init is not None else None
        self.qvel_init = np.array(qvel_init) if qvel_init is not None else None

        super().__init__(env)

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              key: Any = None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the init state handler with its state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        if self.qpos_init is not None:
            data = self.set_qpos(self.qpos_init, data, backend)
        if self.qvel_init is not None:
            data = self.set_qvel(self.qvel_init, data, backend)

        return data, carry

    @staticmethod
    def set_qpos(qpos: Union[np.ndarray, jax.Array],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """
        Set the joint positions in the simulation data.

        Args:
            qpos (List[float]): The joint positions.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        if backend == np:
            data.qpos[:] = qpos
        else:
            data = data.replace(qpos=data.qpos.at[:].set(qpos))

        return data

    @staticmethod
    def set_qvel(qvel: Union[np.ndarray, jax.Array],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """
        Set the joint velocities in the simulation data.

        Args:
            qvel (List[float]): The joint velocities.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        if backend == np:
            data.qvel[:] = qvel
        else:
            data = data.replace(qvel=data.qvel.at[:].set(qvel))

        return data

class CatchInitialStateHandler(DefaultInitialStateHandler):

    """
    Initial state handler for the task of catching a ball.
    When resetting the environment, this handler generates a random root position and orientation for the robot, 
    as well as a random ball spawn position and initial velocity that passes near the robot's midline and chest height.
    The randomization is done in a way that the ball is thrown towards the robot, making it possible for the robot to catch it.

    """

    def __init__(self, env: Any, qpos_init=None, qvel_init=None):
        """
        Initialize the CatchInitialStateHandler class.

        Args:
            env (Any): The environment instance.
            qpos_init (Union[None, List[float]]): Initial joint positions.
            qvel_init (Union[None, List[float]]): Initial joint velocities.
        """

        super().__init__(env, qpos_init=qpos_init, qvel_init=qvel_init)

        # Get the model and info properties from the environment to identify the root and ball joints
        model = env._model
        info_props = env._get_all_info_properties()
        root_jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, info_props["root_free_joint_xml_name"]
        )
        ball_jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, info_props["ball_joint_name"]
        )
        if root_jnt_id < 0 or ball_jnt_id < 0:
            raise ValueError("CatchInitialStateHandler requires root and ball joints in the model")
        self._root_qpos = model.jnt_qposadr[root_jnt_id]
        self._ball_qpos = model.jnt_qposadr[ball_jnt_id]
        self._ball_qvel = model.jnt_dofadr[ball_jnt_id]

        # If the initial joint positions and velocities are not provided, use the default values from the model.
        if self.qpos_init is None:
            self.qpos_init = np.array(model.qpos0, copy=True)
        if self.qvel_init is None:
            self.qvel_init = np.zeros(model.nv)

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              key: Any = None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the init state handler with its state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
            key (Any): Random key for JAX random number generation.

        Returns:
            Tuple[Union[MjData, Data], Any]: The simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        qpos = self.qpos_init.copy() if backend == np else backend.array(self.qpos_init)
        qvel = self.qvel_init.copy() if backend == np else backend.array(self.qvel_init)

        
        # Replace root position and orientation
        if backend == np:
            root_pos = self.random_root_pos()
            ball_pos_vel = self.random_ball_pos_vel(root_pos[:3], root_pos[7])
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            root_pos = self.random_root_pos_jax(key)
            ball_pos_vel = self.random_ball_pos_vel_jax(root_pos[:3], root_pos[7], key)
        if backend == np:
            qpos[self._root_qpos:self._root_qpos+7] = root_pos[:7]
        else:
            qpos = qpos.at[self._root_qpos:self._root_qpos+7].set(root_pos[:7])

        # Replace ball position, orientation and velocity
        if backend == np:
            qpos[self._ball_qpos:self._ball_qpos+7] = ball_pos_vel[:7]
            qvel[self._ball_qvel:self._ball_qvel+3] = ball_pos_vel[7:]
        else:
            qpos = qpos.at[self._ball_qpos:self._ball_qpos+7].set(ball_pos_vel[:7])
            qvel = qvel.at[self._ball_qvel:self._ball_qvel+3].set(ball_pos_vel[7:])
        

        data = self.set_qpos(qpos, data, backend)
        data = self.set_qvel(qvel, data, backend)

        return data, carry

    @staticmethod
    def random_root_pos() -> np.ndarray:
        """
        Generate a random root position and orientation for the robot.

        Returns:
            np.ndarray: A random root position and orientation (as quaternion + angle about z-axis).
        """
        # Randomly sample x and y positions within a specified range
        # x = np.random.uniform(-10, 10)
        # y = np.random.uniform(-10, 10)
        x = 0.0
        y = 0.0
        z = 0.79  # Keep z fixed such that the robot is on the ground

        # Randomly sample a rotation angle around the z-axis
        theta = np.random.uniform(0, 2 * np.pi)

        # convert the rotation angle to a quaternion representation
        qw = np.cos(theta / 2)
        qx = 0
        qy = 0
        qz = np.sin(theta / 2)

        return np.array([x, y, z, qw, qx, qy, qz, theta])

    @staticmethod
    def random_ball_pos_vel(root_position: np.ndarray, root_z_angle: float) -> np.ndarray:
        """
        Generate a ball spawn and initial velocity that pass near the robot's midline and chest height,
        rather than an arc above the torso.

        Returns:
            np.ndarray: A random ball position, orientation (quaternion), and initial velocity.
        """
        # Spawn the ball in front/side of the robot, but aim at a hand-height target near the robot center.
        distance = np.random.uniform(2.5, 7.5)
        angle = np.random.uniform(-np.pi / 4, np.pi / 4) + root_z_angle

        start_x = root_position[0] + distance * np.cos(angle)
        start_y = root_position[1] + distance * np.sin(angle)
        start_z = np.random.uniform(1.0, 1.4)

        target_x = root_position[0] + 0.35 * np.cos(angle)
        target_y = root_position[1] + 0.35 * np.sin(angle)
        target_z = np.random.uniform(0.8, 1.05)

        qw = 1
        qx = 0
        qy = 0
        qz = 0

        g = 9.81
        time_to_reach = np.random.uniform(0.8, 1.4)
        vx = (target_x - start_x) / time_to_reach
        vy = (target_y - start_y) / time_to_reach
        vz = (target_z - start_z) / time_to_reach - 0.5 * g * time_to_reach

        return np.array([start_x, start_y, start_z, qw, qx, qy, qz, vx, vy, vz])

    @staticmethod
    def random_root_pos_jax(key: jax.Array) -> jax.Array:
        """
        Generate a random root position and orientation for the robot.

        Returns:
            jax.Array: A random root position and orientation (as quaternion + angle about z-axis).
        """
        
        position_key, angle_key = jax.random.split(key)
        # randomly sample the x and y coordinates of the robot root position
        # xy = jax.random.uniform(position_key, (2,), minval=-10.0, maxval=10.0)
        # randomly sample the robots rotation about the z-axis
        theta = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        return jnp.array([0.0, 0.0, 0.79, jnp.cos(theta / 2), 0.0,
                          0.0, jnp.sin(theta / 2), theta])

    @staticmethod
    def random_ball_pos_vel_jax(root_position: jax.Array, root_z_angle: jax.Array,
                                key: jax.Array) -> jax.Array:
        """
        Generate a ball spawn and initial velocity that pass near the robot's midline and chest height,
        rather than an arc above the torso.

        Returns:
            jax.Array: A random ball position, orientation (quaternion), and initial velocity.
        """
        
        distance_key, angle_key, height_key, time_key, noise_key = jax.random.split(key, 5)
        # Randomly sample the distance between the robot root and the ball spawn point
        distance = jax.random.uniform(distance_key, (), minval=3.0, maxval=6.0)
        # Randomly sample the angle of the ball spawn point relative to the robot root position
        angle = jax.random.uniform(angle_key, (), minval=-jnp.pi / 4, maxval=jnp.pi / 4) + root_z_angle
        # Randomly sample the height of the ball spawn point
        start_z = jax.random.uniform(height_key, (), minval=1.1, maxval=1.7)
        # Randomly sample the time it takes for the ball to reach the target position
        time_to_reach = jax.random.uniform(time_key, (), minval=0.8, maxval=1.4)

        # Calculate start and target ball position from the sampled data and the robot root position
        start_x = root_position[0] + distance * jnp.cos(angle)
        start_y = root_position[1] + distance * jnp.sin(angle)
        target_x = root_position[0] + 0.35 * jnp.cos(angle)
        target_y = root_position[1] + 0.35 * jnp.sin(angle)
        target_z = jax.random.uniform(noise_key, (), minval=0.8, maxval=1.05)

        # Additive velocity noise
        noise = jax.random.normal(noise_key, (3,)) * jnp.array([0.05, 0.05, 0.05])
        # Calculate the initial velocity of the ball such that it reaches the target position in the sampled time, with added noise
        velocity = jnp.array([
            (target_x - start_x) / time_to_reach,
            (target_y - start_y) / time_to_reach,
            ((target_z - start_z) / time_to_reach) + 0.5 * 9.81 * time_to_reach,
        ]) + noise

        pose = jnp.array([start_x, start_y, start_z, 1.0, 0.0, 0.0, 0.0])
        return jnp.concatenate([pose, velocity])