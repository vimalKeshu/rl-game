import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerEnv


def main() -> None:
    env = CracerEnv(render_mode="human", obs_mode="state", action_mode="discrete")
    obs, info = env.reset()

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required to run the human demo") from exc
    env.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        if info.get("game_mode") == "game_over":
            if keys[pygame.K_SPACE] or keys[pygame.K_r]:
                obs, info = env.reset()
                continue

        action = CracerEnv.action_from_buttons(left, right, up, down)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
