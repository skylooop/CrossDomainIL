"""
Swimmer robot as an explorer in the maze.
Based on `gym`_ (swimmer-v3).
"""

from typing import Tuple

import numpy as np

from envs.mujoco_maze.agent_model import AgentModel
from envs.mujoco_maze.ant import ForwardRewardFn, forward_reward_vnorm


class SwimmerEnv(AgentModel):
    FILE: str = "swimmer.xml"
    MANUAL_COLLISION: bool = True      # default: False
    RADIUS = 0.1  # added

    def __init__(
        self,
        file_path: str = None,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._forward_reward_fn = forward_reward_fn
        super().__init__(file_path, 4)

    def _forward_reward(self, xy_pos_before: np.ndarray) -> Tuple[float, np.ndarray]:
        xy_pos_after = self.sim.data.qpos[:2].copy()
        xy_velocity = (xy_pos_after - xy_pos_before) / self.dt
        return self._forward_reward_fn(xy_velocity)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xy_pos_before = self.sim.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        # forward_reward = self._forward_reward(xy_pos_before)
        # ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        forward_reward = 0
        ctrl_cost = 0
        return (
            self._get_obs(),
            self._forward_reward_weight * forward_reward - ctrl_cost,
            False,
            False,
            dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost),
        )

    def _get_obs(self) -> np.ndarray:
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nv,
        )
        qpos[0] += 1.05
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_xy(self) -> np.ndarray:
        return np.copy(self.sim.data.qpos[:2])


class SwimmerSize3Env(SwimmerEnv):
    def __init__(
        self,
        file_path: str = None,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        super().__init__(file_path, forward_reward_weight, ctrl_cost_weight, forward_reward_fn)

    def viewer_setup(self):
        # size=3
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 13
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 3.
        self.viewer.cam.lookat[1] = 3.
        self.viewer.cam.lookat[2] = 0.


class SwimmerSize4Env(SwimmerEnv):
    def __init__(
            self,
            file_path: str = None,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 1e-4,
            forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        super().__init__(file_path, forward_reward_weight, ctrl_cost_weight, forward_reward_fn)

    def viewer_setup(self):
        # size=4
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 18
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 4.
        self.viewer.cam.lookat[1] = 4.
        self.viewer.cam.lookat[2] = 0.


class SwimmerSize5Env(SwimmerEnv):
    def __init__(
            self,
            file_path: str = None,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 1e-4,
            forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        super().__init__(file_path, forward_reward_weight, ctrl_cost_weight, forward_reward_fn)

    def viewer_setup(self):
        # size=5
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 22.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 5.
        self.viewer.cam.lookat[1] = 5.
        self.viewer.cam.lookat[2] = 0.