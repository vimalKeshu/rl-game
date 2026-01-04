from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerEnv, CracerGymEnv  # noqa: E402
from cracer_sim.env import Config  # noqa: E402


def pygame_available() -> bool:
    try:
        import pygame  # noqa: F401
    except Exception:
        return False
    return True


class TestCracerEnvCore(unittest.TestCase):
    def test_reset_sets_defaults(self) -> None:
        env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=0)
        obs, info = env.reset(seed=0)

        self.assertEqual(env.stage, 1)
        self.assertEqual(env.lives, Config.starting_lives)
        self.assertAlmostEqual(env.fuel, env.max_fuel, places=4)
        self.assertAlmostEqual(env.distance_remaining, Config.stage_distance, places=4)
        self.assertEqual(info["game_mode"], "playing")
        self.assertEqual(len(obs), env.state_size)
        env.close()

    def test_step_reduces_fuel_and_distance(self) -> None:
        env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=1)
        env.reset(seed=1)
        fuel_before = env.fuel
        distance_before = env.distance_remaining

        env.step(0)

        self.assertLess(env.fuel, fuel_before)
        self.assertLess(env.distance_remaining, distance_before)
        env.close()

    def test_game_over_when_fuel_empty(self) -> None:
        env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=2)
        env.reset(seed=2)
        env.fuel = 0.01

        env.step(0)

        self.assertEqual(env.game_mode, "game_over")
        env.close()

    def test_stage_clear_and_advance(self) -> None:
        env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=3)
        env.reset(seed=3)
        env.distance_remaining = 1.0

        env.step(0)
        self.assertEqual(env.game_mode, "stage_clear")
        starting_stage = env.stage

        steps = int(env.fps * 2.1)
        for _ in range(steps):
            env.step(0)

        self.assertEqual(env.stage, starting_stage + 1)
        self.assertEqual(env.game_mode, "playing")
        env.close()

    def test_action_from_buttons(self) -> None:
        self.assertEqual(CracerEnv.action_from_buttons(True, False, False, False), 1)
        self.assertEqual(CracerEnv.action_from_buttons(False, True, False, False), 2)
        self.assertEqual(CracerEnv.action_from_buttons(False, False, True, False), 3)
        self.assertEqual(CracerEnv.action_from_buttons(False, False, False, True), 4)
        self.assertEqual(CracerEnv.action_from_buttons(True, False, True, False), 5)
        self.assertEqual(CracerEnv.action_from_buttons(False, True, True, False), 6)
        self.assertEqual(CracerEnv.action_from_buttons(True, False, False, True), 7)
        self.assertEqual(CracerEnv.action_from_buttons(False, True, False, True), 8)


class TestCracerGymEnv(unittest.TestCase):
    def test_state_observation_shape_and_range(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=0)
        obs, info = env.reset(seed=0)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(obs <= 1.0))
        self.assertTrue(np.all(obs >= -1.0))
        self.assertIn("fuel", info)
        env.close()

    def test_step_return_types(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=0)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)

        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()

    def test_left_right_actions_change_position(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=1)
        obs, _ = env.reset(seed=1)
        x0 = obs[0]

        obs_left, _, _, _, _ = env.step(1)
        self.assertLess(obs_left[0], x0)

        obs, _ = env.reset(seed=1)
        x0 = obs[0]
        obs_right, _, _, _, _ = env.step(2)
        self.assertGreater(obs_right[0], x0)
        env.close()

    def test_accel_and_brake_change_speed(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=2)
        obs, _ = env.reset(seed=2)
        speed0 = obs[1]

        for _ in range(10):
            obs, _, _, _, _ = env.step(3)
        self.assertGreater(obs[1], speed0)

        obs, _ = env.reset(seed=2)
        speed0 = obs[1]
        for _ in range(10):
            obs, _, _, _, _ = env.step(4)
        self.assertLess(obs[1], speed0)
        env.close()

    def test_combo_action_accel_left(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=3)
        obs, _ = env.reset(seed=3)
        x0 = obs[0]
        speed0 = obs[1]

        for _ in range(5):
            obs, _, _, _, _ = env.step(5)

        self.assertLess(obs[0], x0)
        self.assertGreater(obs[1], speed0)
        env.close()

    def test_buttons_action_mode(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="buttons", seed=4)
        obs, _ = env.reset(seed=4)
        x0 = obs[0]
        speed0 = obs[1]

        action = np.array([1, 0, 1, 0], dtype=np.int8)
        obs, _, _, _, _ = env.step(action)

        self.assertLess(obs[0], x0)
        self.assertGreater(obs[1], speed0)
        env.close()

    def test_continuous_action_mode(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="continuous", seed=5)
        obs, _ = env.reset(seed=5)
        x0 = obs[0]
        speed0 = obs[1]

        obs, _, _, _, _ = env.step(np.array([-1.0, 1.0], dtype=np.float32))

        self.assertLess(obs[0], x0)
        self.assertGreater(obs[1], speed0)
        env.close()

    def test_reward_positive(self) -> None:
        env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete", seed=6)
        env.reset(seed=6)
        _, reward, _, _, _ = env.step(0)
        self.assertGreater(reward, 0.0)
        env.close()

    def test_pixel_observation(self) -> None:
        if not pygame_available():
            self.skipTest("pygame not installed")

        env = CracerGymEnv(render_mode="rgb_array", obs_mode="pixels", action_mode="discrete", seed=7)
        obs, _ = env.reset(seed=7)

        self.assertEqual(obs.shape, (env.env.height, env.env.width, 3))
        self.assertEqual(obs.dtype, np.uint8)
        env.close()


if __name__ == "__main__":
    unittest.main()
