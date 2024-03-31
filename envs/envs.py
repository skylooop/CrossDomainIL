import numpy as np
from gymnasium import utils
import gymnasium as gym
from gymnasium.envs.mujoco import mujoco_env
import cv2
import os


class _FrameBufferEnv:
    def __init__(self, past_frames):
        self._initialized = False
        self._past_frames = past_frames

    def _init_buffer(self, im):
        self._im_size = im.shape
        self._reset_buffer()

    def _reset_buffer(self, ):
        # self._frames_buffer = np.zeros([self._past_frames] + list(self._im_size))
        self._frames_buffer = np.zeros([self._past_frames] + list(self._im_size)).astype('uint8')

    def _update_buffer(self, im):
        # self._frames_buffer = np.concatenate([np.expand_dims(im, 0), self._frames_buffer[:-1, :, :, :]], axis=0)
        self._frames_buffer = np.concatenate([np.expand_dims(im.astype('uint8'), 0),
                                              self._frames_buffer[:-1, :, :, :]], axis=0).astype('uint8')


class _CustomInvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }
    
    def __init__(self, size=(32, 32), color_permutation=[0, 1, 2],
                 smoothing_factor=0.0, past_frames=4, not_done=True, render_mode='rgb_array'):
        self._size = size
        self._not_done = not_done
        self._color_permutation = color_permutation
        self._smooth = 1.0 - smoothing_factor
        _FrameBufferEnv.__init__(self, past_frames)
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/inverted_pendulum.xml')
        observation_space = gym.spaces.Box(-np.inf, np.inf, (4, ))
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2, render_mode=render_mode, observation_space=observation_space)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # notdone = np.isfinite(ob['obs']).all() and (np.abs(ob['obs'][1]) <= .2)
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        if done and self._not_done:
            done = False
            reward = 0.0
        return ob, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        # obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        # raw_im = (self.render(mode='rgb_array'))[:, :, self._color_permutation] * self._smooth
        # im = cv2.resize(raw_im, dsize=self._size, interpolation=cv2.INTER_AREA)
        # if not self._initialized:
        #     self._init_buffer(im)
        #     self._initialized = True
        # self._update_buffer(im)
        # curr_frames = self._frames_buffer.astype('int32')
        # return {'obs': obs, 'im': curr_frames}
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
        # v = self.viewer
        # v.cam.trackbodyid = 0
        # v.cam.distance = 2
        # v.cam.lookat[2] = 0.5

    def get_ims(self):
        raw_im = (self.render(mode='rgb_array'))[:, :, self._color_permutation] * self._smooth
        im = cv2.resize(raw_im, dsize=self._size, interpolation=cv2.INTER_AREA)
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class AgentInvertedPendulumEnv(_CustomInvertedPendulumEnv):
    def __init__(self, ):
        super(AgentInvertedPendulumEnv, self).__init__(size=(32, 32),
                                                       color_permutation=[0, 1, 2],
                                                       smoothing_factor=0.0,
                                                       past_frames=4,
                                                       not_done=True)


class ExpertInvertedPendulumEnv(_CustomInvertedPendulumEnv):
    def __init__(self, render_mode='rgb_array'):
        super(ExpertInvertedPendulumEnv, self).__init__(size=(32, 32),
                                                        color_permutation=[2, 1, 0],
                                                        smoothing_factor=0.1,
                                                        render_mode=render_mode,
                                                        past_frames=4,
                                                        not_done=True)


class _CustomInvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(self, size=(32, 32), color_permutation=[0, 1, 2],
                 smoothing_factor=0.0, past_frames=4, not_done=True, render_mode = None):
        self._size = size
        self._not_done = not_done
        self._failure = False
        self._color_permutation = color_permutation
        self._smooth = 1.0 - smoothing_factor
        _FrameBufferEnv.__init__(self, past_frames)
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/inverted_double_pendulum.xml')
        observation_space = gym.spaces.Box(-np.inf, np.inf, (11, ))
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2, render_mode=render_mode, observation_space=observation_space)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        if done and self._not_done:
            done = False
            self._failure = True
        if self._failure:
            r = 0.0
        return ob, r, done, False, {}

    def reset_model(self):
        self._failure = False
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.normal(self.model.nv) * .1
        )
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:1],
            np.sin(self.data.qpos[1:]),
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def get_ims(self):
        raw_im = (self.render(mode='rgb_array'))[:, :, self._color_permutation] * self._smooth
        im = cv2.resize(raw_im, dsize=self._size, interpolation=cv2.INTER_AREA)
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames

    # def viewer_setup(self):
    #     v = self.viewer
    #     v.cam.trackbodyid = 0
    #     v.cam.distance = 2
    #     v.cam.lookat[2] = 0.5  # v.model.stat.center[2]


class AgentInvertedDoublePendulumEnv(_CustomInvertedDoublePendulumEnv):
    def __init__(self, render_mode='rgb_array'):
        super(AgentInvertedDoublePendulumEnv, self).__init__(size=(32, 32),
                                                             color_permutation=[0, 1, 2],
                                                             smoothing_factor=0.0,
                                                             render_mode=render_mode,
                                                             past_frames=4,
                                                             not_done=True)


class ExpertInvertedDoublePendulumEnv(_CustomInvertedDoublePendulumEnv):
    def __init__(self, render_mode='rgb_array'):
        super(ExpertInvertedDoublePendulumEnv, self).__init__(size=(32, 32),
                                                              color_permutation=[2, 1, 0],
                                                              smoothing_factor=0.1,
                                                              render_mode=render_mode,
                                                              past_frames=4,
                                                              not_done=True)


class _CustomReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(self, mode='hard', past_frames=4, l2_penalty=False):
        # GlfwContext(offscreen=True)
        _FrameBufferEnv.__init__(self, past_frames)
        self._mode = mode
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/reacher.xml')
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
        return ob, reward, done, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        if self._mode == 'easy':
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([0.1, 0.1])
        elif self._mode == 'normal':
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([-0.1, 0.1])
        elif self._mode == 'hard':
            qpos = self.np_random.uniform(low=-3.0, high=3.0, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([0.1, 0.1])
        elif self._mode == 'orig':
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 0.2:
                    break
        else:
            raise NotImplementedError
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qpos.flat[2:],
            self.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')[160:380, 125:345]
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class TiltedReacherEasyEnv(_CustomReacherEnv):
    def __init__(self, past_frames=4):
        super(TiltedReacherEasyEnv, self).__init__(mode='easy', past_frames=past_frames)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -30.8999998569


class ReacherEasyEnv(_CustomReacherEnv):
    def __init__(self, past_frames=4):
        super(ReacherEasyEnv, self).__init__(mode='easy', past_frames=past_frames)


class _Custom3ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(self, mode='hard', past_frames=4, l2_penalty=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._mode = mode
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/reacher_3_link.xml')
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
        return ob, reward, done, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        if self._mode == 'easy':
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([0.1, 0.1])
        elif self._mode == 'normal':
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([-0.1, 0.1])
        elif self._mode == 'hard':
            qpos = self.np_random.uniform(low=-3.0, high=3.0, size=self.model.nq) + self.init_qpos.flat
            self.goal = np.array([0.1, 0.1])
        elif self._mode == 'orig':
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 0.2:
                    break
        else:
            raise NotImplementedError
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qpos.flat[3:],
            self.data.qvel.flat[:3],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')[160:380, 125:345]
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class ThreeReacherEasyEnv(_Custom3ReacherEnv):
    def __init__(self, past_frames=4):
        super(ThreeReacherEasyEnv, self).__init__(mode='easy', past_frames=past_frames)


class Tilted3ReacherEasyEnv(_Custom3ReacherEnv):
    def __init__(self, past_frames=4):
        super(Tilted3ReacherEasyEnv, self).__init__(mode='easy', past_frames=past_frames)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -30.8999998569


# No Checker
class _CustomHalfCheetahNCEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    
    def __init__(self, past_frames=4, action_penalties=True, reward_xmove_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self._action_penalties = action_penalties
        self._reward_xmove_only = reward_xmove_only
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/half_cheetah_nc.xml')
        observation_space = gym.spaces.Box(-np.inf, np.inf, (17, ))
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2, observation_space=observation_space, render_mode='rgb_array')
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return _CustomHalfCheetahNCEnv(reward_xmove_only=self._reward_xmove_only)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        if not self._reward_xmove_only:
            if self._action_penalties:
                reward_ctrl = - 0.1 * np.square(action).sum()
            else:
                reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        else:
            reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class ExpertHalfCheetahNCEnv(_CustomHalfCheetahNCEnv):
    def __init__(self, past_frames=4, reward_xmove_only=False):
        super(ExpertHalfCheetahNCEnv, self).__init__(past_frames=past_frames, reward_xmove_only=reward_xmove_only)


class _CustomLLHalfCheetahNCEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(self, past_frames=4, action_penalties=True, reward_xmove_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self._action_penalties = action_penalties
        self._reward_xmove_only = reward_xmove_only
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/half_cheetah_locked_legs_nc.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return _CustomLLHalfCheetahNCEnv(reward_xmove_only=self._reward_xmove_only)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        if not self._reward_xmove_only:
            if self._action_penalties:
                reward_ctrl = - 0.1 * np.square(action).sum()
            else:
                reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        else:
            reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class LockedLegsHalfCheetahNCEnv(_CustomLLHalfCheetahNCEnv):
    def __init__(self, past_frames=4, reward_xmove_only=False):
        super(LockedLegsHalfCheetahNCEnv, self).__init__(past_frames=past_frames, reward_xmove_only=reward_xmove_only)
