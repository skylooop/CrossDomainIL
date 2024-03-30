import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import cv2
import os
from envs.envs import _FrameBufferEnv


class _CustomReacher2Env(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, l2_penalty=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/custom_reacher.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        if self._l2_penalty:
            reward_ctrl = - np.mean(np.square(a)) * 2
        else:
            reward_ctrl = 0.0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        seed = np.random.randint(16)
        mag, ang = 0.15 + 0.05 * divmod(seed, 8)[0], divmod(seed, 8)[1] * np.pi / 4.0
        self.goal = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class TiltedCustomReacher2Env(_CustomReacher2Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(TiltedCustomReacher2Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class CustomReacher2Env(_CustomReacher2Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(CustomReacher2Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class _CustomReacher3Env(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, l2_penalty=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/custom_reacher_3_link.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        if self._l2_penalty:
            reward_ctrl = - np.mean(np.square(a)) * 2
        else:
            reward_ctrl = 0.0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        seed = np.random.randint(16)
        mag, ang = 0.15 + 0.05 * divmod(seed, 8)[0], divmod(seed, 8)[1] * np.pi / 4.0
        self.goal = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[3:],
            self.sim.data.qvel.flat[:3],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class CustomReacher3Env(_CustomReacher3Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(CustomReacher3Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class TiltedCustomReacher3Env(_CustomReacher3Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(TiltedCustomReacher3Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
