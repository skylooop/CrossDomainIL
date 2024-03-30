from dmc2gym.wrappers import DMCWrapper, _flatten_obs
import numpy as np
from gym import utils
import cv2
from envs.envs import _FrameBufferEnv


class _DMCWrapper(DMCWrapper):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs,
        visualize_reward,
        from_pixels,
        height,
        width,
        camera_id,
        frame_skip,
        environment_kwargs,
        channels_first
    ):
        super(_DMCWrapper, self).__init__(domain_name=domain_name,
                                          task_name=task_name,
                                          task_kwargs=task_kwargs,
                                          visualize_reward=visualize_reward,
                                          from_pixels=from_pixels,
                                          height=height,
                                          width=width,
                                          camera_id=camera_id,
                                          frame_skip=frame_skip,
                                          environment_kwargs=environment_kwargs,
                                          channels_first=channels_first)

    def step(self, action):
        # assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        # assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        if self._initialized:
            self._reset_buffer()
        return obs


class DMCartPoleBalanceEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        # task_kwargs['time_limit'] = 1000
        super(DMCartPoleBalanceEnv, self).__init__(domain_name='cartpole',
                                                   task_name='balance_sparse',
                                                   task_kwargs=task_kwargs,
                                                   visualize_reward=False,
                                                   from_pixels=False,
                                                   height=64,
                                                   width=64,
                                                   camera_id=0,
                                                   frame_skip=1,
                                                   environment_kwargs=None,
                                                   channels_first=False
                                                   )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        im = self.render(mode='rgb_array')[8:56, 8:56]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMCartPoleSwingUpEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMCartPoleSwingUpEnv, self).__init__(domain_name='cartpole',
                                                   task_name='swingup',
                                                   task_kwargs=task_kwargs,
                                                   visualize_reward=False,
                                                   from_pixels=False,
                                                   height=64,
                                                   width=64,
                                                   camera_id=0,
                                                   frame_skip=1,
                                                   environment_kwargs=None,
                                                   channels_first=False
                                                   )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        self._physics.data.cam_xpos[0][0] = self._physics.get_state()[0]    # track cart x position
        im = self.render(mode='rgb_array')[8:56, 8:56]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMPendulumEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMPendulumEnv, self).__init__(domain_name='pendulum',
                                            task_name='swingup',
                                            task_kwargs=task_kwargs,
                                            visualize_reward=False,
                                            from_pixels=False,
                                            height=64,
                                            width=64,
                                            camera_id=0,
                                            frame_skip=1,
                                            environment_kwargs=None,
                                            channels_first=False
                                            )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        self._physics.data.cam_xpos[0][2] = 0.6
        self._physics.data.cam_xmat[0] = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMAcrobotEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMAcrobotEnv, self).__init__(domain_name='acrobot',
                                           task_name='swingup',
                                           task_kwargs=task_kwargs,
                                           visualize_reward=False,
                                           from_pixels=False,
                                           height=64,
                                           width=64,
                                           camera_id=0,
                                           frame_skip=1,
                                           environment_kwargs=None,
                                           channels_first=False
                                           )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        im = self.render(mode='rgb_array')[4:60, 4:60]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMHopperEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, reward_by_xspeed_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMHopperEnv, self).__init__(domain_name='hopper',
                                          task_name='hop',
                                          task_kwargs=task_kwargs,
                                          visualize_reward=False,
                                          from_pixels=False,
                                          height=64,
                                          width=64,
                                          camera_id=0,
                                          frame_skip=1,
                                          environment_kwargs=None,
                                          channels_first=False
                                          )
        utils.EzPickle.__init__(self)
        self.reward_by_xspeed_only = reward_by_xspeed_only
        if self.reward_by_xspeed_only:
            print("\n\033[91m" + "Warning: Reward is only based on agent's speed along x-axis." + "\033[0m")

    def __deepcopy__(self, memodict={}):
        return DMHopperEnv(reward_by_xspeed_only=self.reward_by_xspeed_only)

    def step(self, action):
        # assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        # assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if self.reward_by_xspeed_only:
                reward = self.physics.speed() * 0.001       # average speed along x-axis (global coord.)
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMWalkerEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, reward_by_xspeed_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMWalkerEnv, self).__init__(domain_name='walker',
                                          task_name='run',
                                          task_kwargs=task_kwargs,
                                          visualize_reward=False,
                                          from_pixels=False,
                                          height=64,
                                          width=64,
                                          camera_id=0,
                                          frame_skip=1,
                                          environment_kwargs=None,
                                          channels_first=False
                                          )
        utils.EzPickle.__init__(self)
        self.reward_by_xspeed_only = reward_by_xspeed_only
        if self.reward_by_xspeed_only:
            print("\n\033[91m" + "Warning: Reward is only based on agent's speed along x-axis." + "\033[0m")

    def __deepcopy__(self, memodict={}):
        return DMWalkerEnv(reward_by_xspeed_only=self.reward_by_xspeed_only)

    def step(self, action):
        # assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        # assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if self.reward_by_xspeed_only:
                reward = self.physics.horizontal_velocity() * 0.001     # average speed along x-axis (global coord.)
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMCheetahEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, reward_by_xspeed_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMCheetahEnv, self).__init__(domain_name='cheetah',
                                          task_name='run',
                                          task_kwargs=task_kwargs,
                                          visualize_reward=False,
                                          from_pixels=False,
                                          height=64,
                                          width=64,
                                          camera_id=0,
                                          frame_skip=1,
                                          environment_kwargs=None,
                                          channels_first=False
                                          )
        utils.EzPickle.__init__(self)
        self.reward_by_xspeed_only = reward_by_xspeed_only
        if self.reward_by_xspeed_only:
            print("\n\033[91m" + "Warning: Reward is only based on agent's speed along x-axis." + "\033[0m")

    def __deepcopy__(self, memodict={}):
        return DMCheetahEnv(reward_by_xspeed_only=self.reward_by_xspeed_only)

    def step(self, action):
        # assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        # assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if self.reward_by_xspeed_only:
                reward = self.physics.speed() * 0.001   # average speed along x-axis (global coord.)
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames