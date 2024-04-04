"""
A ball-like robot as an explorer in the maze.
Based on `models`_ and `rllab`_.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import mujoco

from envs.mujoco_maze.agent_model import AgentModel

class PointEnv(AgentModel):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    FILE: str = "point.xml"
    ORI_IND: int = 2
    MANUAL_COLLISION: bool = True
    RADIUS: float = 1
    OBJBALL_TYPE: str = "hinge"

    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: Optional[str] = None, **kwargs) -> None:
        default_camera_config = {"distance": 11, "elevation": -90, "lookat": np.array((3, 3, 3))}
        high = np.inf * np.ones(6, dtype=np.float32)
        high[3:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        observation_space = gym.spaces.Box(low, high)
        super().__init__(file_path, default_camera_config, 1, observation_space, **kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        qpos = self.data.qpos.copy()
        qpos[2] += action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction
        qpos[0] += np.cos(ori) * action[0]
        qpos[1] += np.sin(ori) * action[0]
        qvel = np.clip(self.data.qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        next_obs = self._get_obs()
        return next_obs, 0.0, False, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:3],  # Only point-relevant coords.
                self.data.qvel.flat[:3],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.normal(self.model.nv) * 0.1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_xy(self):
        return self.data.qpos[:2].copy()

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.data.qvel)

    def get_ori(self):
        return self.data.qpos[self.ORI_IND]


class PointSize3Env(PointEnv):
    def __init__(self, file_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(file_path, **kwargs)

    def viewer_setup(self):
        # size=3
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 15.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 3.
        self.viewer.cam.lookat[1] = 1.
        self.viewer.cam.lookat[2] = 0.


class PointSize4Env(PointEnv):
    def __init__(self, file_path: Optional[str] = None) -> None:
        super().__init__(file_path)

    def viewer_setup(self):
        # size=4
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 18
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 4.
        self.viewer.cam.lookat[1] = 4.
        self.viewer.cam.lookat[2] = 0.


class PointSize5Env(PointEnv):
    def __init__(self, file_path: Optional[str] = None) -> None:
        super().__init__(file_path)

    def viewer_setup(self):
        # size=5
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 22.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 5.
        self.viewer.cam.lookat[1] = 5.
        self.viewer.cam.lookat[2] = 0.
