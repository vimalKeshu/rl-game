from __future__ import annotations

from typing import Optional, Any


class PygameRenderer:
    def __init__(self, width: int, height: int, mode: str) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required for rendering") from exc

        self.pygame = pygame
        self.width = width
        self.height = height
        self.mode = mode

        pygame.init()
        pygame.font.init()

        self.screen = None
        if mode == "human":
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Cracer Simulation")
            self.surface = self.screen
        else:
            self.surface = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.big_font = pygame.font.Font(None, 36)

    def close(self) -> None:
        self.pygame.quit()

    def tick(self, fps: int) -> None:
        if self.mode == "human":
            self.clock.tick(fps)

    def draw(self, env) -> Optional[Any]:
        pg = self.pygame
        surface = self.surface

        surface.fill((28, 97, 36))

        road_center_x = env.width / 2.0 + env.road_offset
        road_center_y = env.height / 2.0
        road_rect = self._rect_from_center(road_center_x, road_center_y, env.road_width, env.road_height)
        pg.draw.rect(surface, (46, 46, 46), road_rect)

        edge_width = 6
        left_edge_x = road_center_x - env.road_width / 2.0 + edge_width / 2.0
        right_edge_x = road_center_x + env.road_width / 2.0 - edge_width / 2.0
        left_rect = self._rect_from_center(left_edge_x, road_center_y, edge_width, env.road_height)
        right_rect = self._rect_from_center(right_edge_x, road_center_y, edge_width, env.road_height)
        pg.draw.rect(surface, (250, 219, 56), left_rect)
        pg.draw.rect(surface, (250, 219, 56), right_rect)

        marker_w, marker_h = 8, 40
        for marker in env.lane_markers:
            x = road_center_x + marker[0]
            y = road_center_y + marker[1]
            rect = self._rect_from_center(x, y, marker_w, marker_h)
            pg.draw.rect(surface, (242, 242, 242), rect)

        for entity in env.hazards:
            self._draw_entity(surface, env, entity)
        for entity in env.pickups:
            self._draw_entity(surface, env, entity)
        for entity in env.enemies:
            self._draw_entity(surface, env, entity)

        self._draw_player(surface, env)
        self._draw_hud(surface, env)

        if self.mode == "human":
            pg.display.flip()
            return None
        return self._get_rgb_array()

    def _draw_player(self, surface, env) -> None:
        pg = self.pygame
        player_color = (235, 38, 46)
        if env.invincible_time > 0.0 and int(env.invincible_time * 12) % 2 == 0:
            player_color = (160, 40, 40)
        rect = self._rect_from_center(env.player_x, env.player_y, env.player_width, env.player_height)
        pg.draw.rect(surface, player_color, rect)

        stripe = self._rect_from_center(env.player_x, env.player_y, env.player_width * 0.2, env.player_height * 0.9)
        pg.draw.rect(surface, (255, 255, 255), stripe)

        windshield = self._rect_from_center(
            env.player_x,
            env.player_y + env.player_height * 0.15,
            env.player_width * 0.6,
            env.player_height * 0.2,
        )
        pg.draw.rect(surface, (153, 230, 255), windshield)

    def _draw_entity(self, surface, env, entity) -> None:
        pg = self.pygame
        if entity.kind == "truck":
            color = (51, 128, 219)
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, color, rect)
            cab = self._rect_from_center(
                entity.x,
                entity.y + entity.height * 0.26,
                entity.width * 0.9,
                entity.height * 0.28,
            )
            pg.draw.rect(surface, (25, 25, 25), cab)
            stripe = self._rect_from_center(
                entity.x,
                entity.y - entity.height * 0.05,
                entity.width * 0.18,
                entity.height * 0.85,
            )
            pg.draw.rect(surface, (230, 230, 230), stripe)
        elif entity.kind == "race":
            color = (242, 51, 140)
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, color, rect)
            visor = self._rect_from_center(
                entity.x,
                entity.y + entity.height * 0.18,
                entity.width * 0.6,
                entity.height * 0.16,
            )
            pg.draw.rect(surface, (153, 230, 255), visor)
            stripe = self._rect_from_center(entity.x, entity.y, entity.width * 0.18, entity.height * 0.9)
            pg.draw.rect(surface, (242, 242, 242), stripe)
        elif entity.kind == "enemy":
            color = (242, 194, 51)
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, color, rect)
            windshield = self._rect_from_center(
                entity.x,
                entity.y + entity.height * 0.12,
                entity.width * 0.55,
                entity.height * 0.18,
            )
            pg.draw.rect(surface, (179, 230, 255), windshield)
            stripe = self._rect_from_center(entity.x, entity.y, entity.width * 0.2, entity.height * 0.85)
            pg.draw.rect(surface, (25, 25, 25), stripe)
        elif entity.kind == "fuel":
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, (250, 133, 31), rect)
            cap = self._rect_from_center(
                entity.x,
                entity.y + entity.height * 0.35,
                entity.width * 0.4,
                entity.height * 0.15,
            )
            pg.draw.rect(surface, (10, 10, 10), cap)
        elif entity.kind == "pothole":
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, (13, 13, 13), rect)
            pg.draw.ellipse(surface, (90, 90, 90), rect, width=2)
        elif entity.kind == "bump":
            rect = self._rect_from_center(entity.x, entity.y, entity.width, entity.height)
            pg.draw.rect(surface, (128, 64, 20), rect)
            highlight = self._rect_from_center(
                entity.x,
                entity.y + entity.height * 0.2,
                entity.width * 0.85,
                entity.height * 0.35,
            )
            pg.draw.rect(surface, (220, 220, 220), highlight)

    def _draw_hud(self, surface, env) -> None:
        pg = self.pygame
        margin = 16

        fuel_label = self.font.render("FUEL", True, (255, 255, 255))
        surface.blit(fuel_label, (margin, margin))

        bar_x = margin
        bar_y = margin + 18
        bar_w = 150
        bar_h = 10
        pg.draw.rect(surface, (30, 30, 30), (bar_x, bar_y, bar_w, bar_h))
        fuel_ratio = max(0.0, min(1.0, env.fuel / env.max_fuel))
        if fuel_ratio > 0.6:
            fuel_color = (51, 204, 51)
        elif fuel_ratio > 0.3:
            fuel_color = (230, 179, 51)
        else:
            fuel_color = (230, 51, 51)
        pg.draw.rect(surface, fuel_color, (bar_x, bar_y, int(bar_w * fuel_ratio), bar_h))

        lives_text = self.font.render(f"LIVES {env.lives}", True, (255, 255, 255))
        surface.blit(lives_text, (margin, bar_y + 16))

        right_x = env.width - margin
        speed_text = self.font.render(
            f"SPEED {int(env.current_speed)} / LIMIT {int(env.speed_limit)}", True, (255, 255, 255)
        )
        distance_text = self.font.render(f"GOAL {int(env.distance_remaining)}", True, (255, 255, 255))
        stage_text = self.font.render(f"STAGE {env.stage}", True, (255, 255, 255))
        score_text = self.font.render(f"SCORE {int(env.score)}", True, (255, 255, 255))

        surface.blit(speed_text, (right_x - speed_text.get_width(), margin))
        surface.blit(distance_text, (right_x - distance_text.get_width(), margin + 18))
        surface.blit(stage_text, (right_x - stage_text.get_width(), margin + 36))
        surface.blit(score_text, (right_x - score_text.get_width(), margin + 54))

        message = env.current_message()
        if message:
            msg_surf = self.big_font.render(message, True, (255, 255, 255))
            surface.blit(
                msg_surf,
                (
                    (env.width - msg_surf.get_width()) / 2.0,
                    (env.height - msg_surf.get_height()) / 2.0,
                ),
            )

        hint = self.font.render("ARROWS/WASD: STEER + SPEED   SPACE: RESTART", True, (220, 220, 220))
        surface.blit(hint, (margin, env.height - 28))

    def _get_rgb_array(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("rgb_array mode requires numpy") from exc

        arr = self.pygame.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2))

    def _rect_from_center(self, x: float, y: float, w: float, h: float):
        return self.pygame.Rect(
            int(x - w / 2.0),
            int(self.height - y - h / 2.0),
            int(w),
            int(h),
        )
