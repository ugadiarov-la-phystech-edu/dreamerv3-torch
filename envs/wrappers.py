import datetime
import gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SlotObservation(gym.Wrapper):
    def __init__(self, env, slot_shape):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        del spaces["image"]
        spaces["slots"] = gym.spaces.Box(-np.inf, np.inf, shape=slot_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()


class SlotsWrapper:
    def __init__(self, envs, slot_extractor, slots_shape):
        self._envs = envs
        self._slot_extractor = slot_extractor
        self._slots_shape = slots_shape
        self._prev_slots = np.zeros((len(self._envs), *slots_shape), dtype=np.float32)

    def __len__(self):
        return len(self._envs)

    def get_env_id(self, index):
        return self._envs[index].id

    @staticmethod
    def match(masks, masks_predict):
        not_matched_masks = set(range(masks_predict.shape[0]))
        not_matched_masks_predict = set(range(masks_predict.shape[0]))
        matching = [None] * masks_predict.shape[0]
        for i in range(len(masks)):
            mask = masks[i]
            area = mask.sum()
            if area == 0:
                continue

            best_match_index = None
            best_match_score = 0
            for j in not_matched_masks_predict:
                score = masks_predict[j][mask].sum() / area
                if score > best_match_score:
                    best_match_index = j
                    best_match_score = score

            matching[i] = best_match_index
            not_matched_masks.remove(i)
            not_matched_masks_predict.remove(best_match_index)

        for i, j in zip(not_matched_masks, not_matched_masks_predict):
            matching[i] = j

        return matching

    def reset(self, indices):
        results = [self._envs[i].reset() for i in indices]
        results = [r() for r in results]
        images = np.stack([result.pop("image") for result in results])
        batch_masks = [result.pop("masks") for result in results]
        slots, batch_masks_predict = self._slot_extractor(images, prev_slots=None, to_numpy=True)

        matching = []
        for masks, masks_predict in zip(batch_masks, batch_masks_predict):
            matching.append(self.match(masks, masks_predict))

        matching = np.asarray(matching)
        slots = np.take_along_axis(slots, matching[..., np.newaxis], axis=1)
        self._prev_slots[indices] = slots

        for i, result in enumerate(results):
            result["vector"] = slots[i].reshape(-1)

        return results

    def step(self, action):
        assert len(action) == len(self)
        results = [e.step(a) for e, a in zip(self._envs, action)]
        results = [r() for r in results]
        images = np.stack([result[0].pop("image") for result in results])
        batch_masks = [result[0].pop("masks") for result in results]
        slots, batch_masks_predict = self._slot_extractor(images, prev_slots=self._prev_slots, to_numpy=True)

        matching = []
        for masks, masks_predict in zip(batch_masks, batch_masks_predict):
            matching.append(self.match(masks, masks_predict))

        matching = np.asarray(matching)
        slots = np.take_along_axis(slots, matching[..., np.newaxis], axis=1)

        self._prev_slots[:] = slots
        for result, slot in zip(results, slots):
            result[0]["vector"] = slot.reshape(-1)

        return results

    def get_action_space(self):
        return self._envs[0].action_space

    def get_observation_space(self):
        return self._envs[0].observation_space

    def close(self):
        for env in self._envs:
            env.close()
