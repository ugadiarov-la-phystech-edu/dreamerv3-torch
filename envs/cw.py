import numpy as np

from envs.cw_envs import CwTargetEnv


class CwEnv:
    metadata = {}

    def __init__(self, env_config, seed=0):
        self._env = CwTargetEnv(env_config, seed)
        self._env.action_space.seed(seed)
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        reward = np.float32(reward)
        unwrapped = self._env.unwrapped
        obs = {
            "image": image,
            "is_first": False,
            "is_last": done,
            "is_terminal":  done and unwrapped._episode_length <= unwrapped._max_episode_length
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self):
        image = self._env.reset()
        obs = {
            "image": image,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
