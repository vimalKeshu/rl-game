from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise RuntimeError("gymnasium is required to use CracerGymEnv") from exc

from .env import CracerEnv


class CracerGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        obs_mode: str = "state",
        action_mode: str = "discrete",
        width: int = 800,
        height: int = 600,
        max_objects: int = 6,
        fps: int = 60,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.env = CracerEnv(
            width=width,
            height=height,
            fps=fps,
            render_mode=render_mode,
            obs_mode=obs_mode,
            action_mode=action_mode,
            seed=seed,
            max_objects=max_objects,
        )
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.action_mode = action_mode

        self.action_space = self._make_action_space(action_mode)
        self.observation_space = self._make_observation_space(obs_mode)
        self.metadata["render_fps"] = fps

    def _make_action_space(self, action_mode: str):
        if action_mode == "discrete":
            return spaces.Discrete(9)
        if action_mode == "continuous":
            return spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        if action_mode == "buttons":
            return spaces.MultiBinary(4)
        raise ValueError(f"Unknown action_mode: {action_mode}")

    def _make_observation_space(self, obs_mode: str):
        if obs_mode == "state":
            size = self.env.state_size
            return spaces.Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)
        if obs_mode in ("pixels", "rgb_array"):
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.env.height, self.env.width, 3),
                dtype=np.uint8,
            )
        raise ValueError(f"Unknown obs_mode: {obs_mode}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        obs, info = self.env.reset(seed=seed)
        return self._format_obs(obs), info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        if self.action_mode == "buttons" and not isinstance(action, dict):
            action = self._buttons_from_array(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render(mode=self.render_mode)

    def close(self) -> None:
        self.env.close()

    def _format_obs(self, obs: Any):
        if self.obs_mode == "state":
            return np.asarray(obs, dtype=np.float32)
        if self.obs_mode in ("pixels", "rgb_array"):
            if obs is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return obs.astype(np.uint8, copy=False)
        return obs

    @staticmethod
    def _buttons_from_array(action: Any) -> dict:
        try:
            left, right, accelerate, brake = action
        except (TypeError, ValueError):
            left = right = accelerate = brake = 0
        return {
            "left": bool(left),
            "right": bool(right),
            "accelerate": bool(accelerate),
            "brake": bool(brake),
        }
