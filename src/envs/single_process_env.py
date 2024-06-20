from typing import Any, Tuple

import numpy as np

from .done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> Tuple[np.ndarray, Any]:
        self.reset_done_tracker()
        obs, info = self.env.reset()
        return obs[None, ...], info

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, reward, done, trunc, _ = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        done = np.array([done])
        trunc = np.array([trunc])
        self.update_done_tracker(done | trunc)
        return obs[None, ...], np.array([reward]), done, trunc, None

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
