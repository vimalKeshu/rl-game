"""Pygame front end for Cancer Kingdom.

Run from this directory with: python3 game.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:  # pragma: no cover - only exercised without pygame
    raise SystemExit("pygame is required. Install it with: pip install -r requirements.txt") from exc

from simulation import CancerKingdom, TREATMENTS


WIDTH, HEIGHT = 1280, 800
MAP_RIGHT = 830

INK = (225, 235, 244)
MUTED = (135, 154, 169)
BG = (7, 13, 22)
PANEL = (13, 24, 36)
PANEL_LIGHT = (20, 34, 48)
LINE = (39, 59, 73)
CYAN = (47, 207, 196)
BLUE = (54, 132, 210)
GREEN = (63, 190, 120)
YELLOW = (240, 184, 65)
ORANGE = (237, 116, 50)
RED = (226, 67, 83)
MAGENTA = (204, 69, 139)


ORGAN_POSITIONS: Dict[str, Tuple[int, int]] = {
    "Brain": (414, 145),
    "Lungs": (285, 275),
    "Lymph Nodes": (540, 275),
    "Liver": (285, 430),
    "Stomach": (540, 430),
    "Kidneys": (285, 575),
    "Bone Marrow": (540, 575),
    "Bone": (700, 430),
}


class Button:
    def __init__(self, rect: pygame.Rect, key: str) -> None:
        self.rect = rect
        self.key = key


class CancerKingdomGame:
    def __init__(self, seed: Optional[int] = None) -> None:
        pygame.init()
        pygame.display.set_caption("Cancer Kingdom — Evolutionary Strategy")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.sim = CancerKingdom(seed=seed)
        self.selected = "Lungs"
        self.running = True
        self.show_help = True
        self.toast = ""
        self.toast_until = 0
        self.mouse = (0, 0)

        self.font_xs = pygame.font.SysFont("arial", 13)
        self.font_sm = pygame.font.SysFont("arial", 15)
        self.font_md = pygame.font.SysFont("arial", 18)
        self.font_lg = pygame.font.SysFont("arial", 26, bold=True)
        self.font_title = pygame.font.SysFont("arial", 36, bold=True)
        self.font_huge = pygame.font.SysFont("arial", 48, bold=True)

        keys = list(TREATMENTS)
        self.treatment_buttons: List[Button] = [
            Button(pygame.Rect(854, 318 + index * 58, 400, 50), key)
            for index, key in enumerate(keys)
        ]
        self.end_button = pygame.Rect(854, 684, 400, 58)
        self.help_button = pygame.Rect(1202, 24, 48, 32)

    def run(self) -> None:
        while self.running:
            self.mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                self.handle_event(event)
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.show_help:
                    self.show_help = False
                else:
                    self.running = False
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_r:
                self.restart()
            elif event.key == pygame.K_SPACE and not self.show_help:
                self.sim.end_turn()
            elif pygame.K_1 <= event.key <= pygame.K_6 and not self.show_help:
                key = list(TREATMENTS)[event.key - pygame.K_1]
                self.apply_treatment(key)
            return
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        if self.show_help:
            self.show_help = False
            return
        if self.sim.status != "playing":
            if pygame.Rect(490, 500, 300, 56).collidepoint(event.pos):
                self.restart()
            return
        if self.help_button.collidepoint(event.pos):
            self.show_help = True
            return
        for name, center in ORGAN_POSITIONS.items():
            if math.dist(center, event.pos) <= 54:
                self.selected = name
                return
        for button in self.treatment_buttons:
            if button.rect.collidepoint(event.pos):
                self.apply_treatment(button.key)
                return
        if self.end_button.collidepoint(event.pos):
            self.sim.end_turn()

    def restart(self) -> None:
        self.sim.reset()
        self.selected = "Lungs"
        self.show_help = False
        self.toast = "New patient started"
        self.toast_until = pygame.time.get_ticks() + 1600

    def apply_treatment(self, key: str) -> None:
        ok, reason = self.sim.can_treat(key, self.selected)
        if not ok:
            self.toast = reason
            self.toast_until = pygame.time.get_ticks() + 1800
            return
        self.sim.treat(key, self.selected)
        self.toast = f"{TREATMENTS[key].name} administered"
        self.toast_until = pygame.time.get_ticks() + 1400

    def draw(self) -> None:
        self.screen.fill(BG)
        self.draw_map_panel()
        self.draw_sidebar()
        if pygame.time.get_ticks() < self.toast_until:
            self.draw_toast(self.toast)
        if self.show_help:
            self.draw_help()
        elif self.sim.status != "playing":
            self.draw_game_over()

    def draw_map_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL, (18, 18, 796, 764), border_radius=16)
        pygame.draw.rect(self.screen, LINE, (18, 18, 796, 764), width=1, border_radius=16)

        self.text("CANCER KINGDOM", (42, 34), self.font_lg, INK)
        self.text("EVOLUTIONARY ONCOLOGY COMMAND", (42, 64), self.font_xs, CYAN)

        self.draw_metric(310, 34, "TURN", f"{self.sim.turn}/{self.sim.MAX_TURNS}", BLUE)
        self.draw_metric(430, 34, "BODY HEALTH", f"{self.sim.body_health:.0f}%", GREEN)
        self.draw_metric(585, 34, "TUMOR BURDEN", f"{self.sim.total_cancer:.1f}", RED)

        pygame.draw.line(self.screen, LINE, (42, 93), (790, 93), 1)
        self.draw_body_network()
        self.draw_event_log()

    def draw_metric(self, x: int, y: int, label: str, value: str, color: Tuple[int, int, int]) -> None:
        self.text(label, (x, y), self.font_xs, MUTED)
        self.text(value, (x, y + 18), self.font_md, color)

    def draw_body_network(self) -> None:
        route_seen = set()
        for start, destinations in self.sim.ROUTES.items():
            for destination in destinations:
                edge = tuple(sorted((start, destination)))
                if edge in route_seen:
                    continue
                route_seen.add(edge)
                a = ORGAN_POSITIONS[start]
                b = ORGAN_POSITIONS[destination]
                cancer_on_route = self.sim.organs[start].cancer > 0.1 and self.sim.organs[destination].cancer > 0.1
                pygame.draw.line(self.screen, (75, 36, 53) if cancer_on_route else (32, 58, 70), a, b, 3)
                pygame.draw.line(self.screen, (34, 112, 126), a, b, 1)

        pulse = (math.sin(pygame.time.get_ticks() / 270.0) + 1.0) / 2.0
        for name, center in ORGAN_POSITIONS.items():
            organ = self.sim.organs[name]
            selected = name == self.selected
            radius = 51
            if selected:
                ring_radius = int(58 + pulse * 3)
                pygame.draw.circle(self.screen, (*CYAN, 40), center, ring_radius, width=2)

            health_ratio = organ.health / 100.0
            organ_color = mix(RED, (41, 117, 113), health_ratio)
            pygame.draw.circle(self.screen, (5, 11, 18), center, radius + 4)
            pygame.draw.circle(self.screen, organ_color, center, radius)
            pygame.draw.circle(self.screen, mix((65, 84, 92), INK, health_ratio), center, radius, width=2)

            self.draw_tumor_cells(name, center, radius)
            label_width = self.font_sm.size(name)[0]
            label_rect = pygame.Rect(center[0] - label_width // 2 - 7, center[1] + 59, label_width + 14, 23)
            pygame.draw.rect(self.screen, (9, 18, 28), label_rect, border_radius=5)
            self.text(name, (center[0] - label_width // 2, center[1] + 62), self.font_sm, INK)
            self.text(
                f"{organ.cancer:.1f}",
                (center[0], center[1] - 10),
                self.font_md,
                (255, 240, 240) if organ.cancer else INK,
                center=True,
            )
            self.text("BURDEN", (center[0], center[1] + 11), self.font_xs, (235, 195, 205), center=True)

        self.text(self.sim.progression, (42, 108), self.font_sm, YELLOW)
        self.text("Blood / lymphatic routes", (626, 109), self.font_xs, MUTED)

    def draw_tumor_cells(self, name: str, center: Tuple[int, int], radius: int) -> None:
        organ = self.sim.organs[name]
        count = min(38, int(organ.cancer * 0.8))
        if organ.cancer > 0.05:
            count = max(2, count)
        seed = sum((i + 1) * ord(char) for i, char in enumerate(name))
        for index in range(count):
            angle = ((seed * 0.11 + index * 2.399) % (math.pi * 2))
            distance = 15 + ((seed + index * 37) % max(1, radius - 21))
            x = center[0] + int(math.cos(angle) * distance)
            y = center[1] + int(math.sin(angle) * distance)
            resistant = index < round(count * organ.resistance / 100.0)
            color = MAGENTA if resistant else RED
            pygame.draw.circle(self.screen, color, (x, y), 3 if resistant else 2)

    def draw_event_log(self) -> None:
        rect = pygame.Rect(40, 670, 752, 90)
        pygame.draw.rect(self.screen, (9, 18, 28), rect, border_radius=10)
        self.text("BIOLOGICAL EVENT LOG", (54, 681), self.font_xs, CYAN)
        events = self.sim.log[-3:]
        for index, event in enumerate(reversed(events)):
            color = INK if index == 0 else MUTED
            self.text("• " + shorten(event, 95), (54, 706 + index * 18), self.font_xs, color)

    def draw_sidebar(self) -> None:
        pygame.draw.rect(self.screen, PANEL, (830, 18, 432, 764), border_radius=16)
        pygame.draw.rect(self.screen, LINE, (830, 18, 432, 764), width=1, border_radius=16)

        self.text("TREATMENT COMMAND", (854, 34), self.font_md, INK)
        pygame.draw.rect(self.screen, PANEL_LIGHT, self.help_button, border_radius=8)
        self.text("H", self.help_button.center, self.font_sm, CYAN, center=True)

        self.text("ACTION POINTS", (854, 77), self.font_xs, MUTED)
        for index in range(self.sim.ACTION_POINTS):
            color = CYAN if index < self.sim.action_points else (41, 57, 67)
            pygame.draw.circle(self.screen, color, (963 + index * 25, 84), 8)

        organ = self.sim.organs[self.selected]
        selected_rect = pygame.Rect(854, 106, 384, 188)
        pygame.draw.rect(self.screen, (10, 20, 31), selected_rect, border_radius=11)
        pygame.draw.rect(self.screen, LINE, selected_rect, 1, border_radius=11)
        self.text(self.selected.upper(), (871, 119), self.font_md, CYAN)
        self.text(f"Tumor burden  {organ.cancer:.1f}", (871, 150), self.font_sm, INK)
        self.draw_bar(871, 173, 348, organ.cancer, 100.0, RED)
        self.stat_pair("Organ health", organ.health, "%", 871, 194, GREEN)
        self.stat_pair("Resistant clones", organ.resistance, "%", 871, 218, MAGENTA)
        self.stat_pair("Blood supply", organ.vascularity, "%", 871, 242, BLUE)
        self.stat_pair("Immune activity", organ.immune, "%", 871, 266, CYAN)

        for index, button in enumerate(self.treatment_buttons):
            self.draw_treatment_button(button, index + 1)

        hovered = self.end_button.collidepoint(self.mouse)
        color = (34, 119, 117) if hovered and self.sim.status == "playing" else (25, 81, 84)
        if self.sim.status != "playing":
            color = (39, 50, 57)
        pygame.draw.rect(self.screen, color, self.end_button, border_radius=10)
        self.text("END TURN", (874, 699), self.font_md, INK)
        self.text("SPACE", (1194, 703), self.font_xs, (181, 216, 215))
        self.text("Cancer grows, adapts, and may spread", (874, 725), self.font_xs, (181, 216, 215))

        self.text("R  Restart", (854, 754), self.font_xs, MUTED)
        self.text("H  Help", (1169, 754), self.font_xs, MUTED)

    def draw_treatment_button(self, button: Button, number: int) -> None:
        treatment = TREATMENTS[button.key]
        allowed, _ = self.sim.can_treat(button.key, self.selected)
        hovered = button.rect.collidepoint(self.mouse)
        if not allowed:
            background, border, text_color = (16, 25, 34), (31, 44, 53), (91, 105, 114)
        elif hovered:
            background, border, text_color = (27, 54, 66), CYAN, INK
        else:
            background, border, text_color = PANEL_LIGHT, LINE, INK
        pygame.draw.rect(self.screen, background, button.rect, border_radius=9)
        pygame.draw.rect(self.screen, border, button.rect, width=1, border_radius=9)
        self.text(str(number), (870, button.rect.y + 16), self.font_xs, MUTED)
        self.text(treatment.name, (894, button.rect.y + 8), self.font_sm, text_color)
        scope = "TARGET" if treatment.targeted else "SYSTEMIC"
        self.text(scope, (894, button.rect.y + 28), self.font_xs, MUTED if allowed else text_color)
        self.text(f"{treatment.cost} AP", (1207, button.rect.y + 16), self.font_sm, CYAN if allowed else text_color)

        if hovered:
            tooltip = pygame.Rect(555, min(button.rect.y, 592), 270, 62)
            pygame.draw.rect(self.screen, (18, 31, 43), tooltip, border_radius=8)
            pygame.draw.rect(self.screen, LINE, tooltip, 1, border_radius=8)
            self.draw_wrapped(treatment.description, tooltip.inflate(-20, -14), self.font_xs, INK, 17)

    def stat_pair(
        self, label: str, value: float, suffix: str, x: int, y: int, color: Tuple[int, int, int]
    ) -> None:
        self.text(label, (x, y), self.font_xs, MUTED)
        self.text(f"{value:.0f}{suffix}", (1218, y), self.font_xs, color, right=True)

    def draw_bar(
        self, x: int, y: int, width: int, value: float, maximum: float, color: Tuple[int, int, int]
    ) -> None:
        pygame.draw.rect(self.screen, (32, 45, 54), (x, y, width, 7), border_radius=4)
        filled = int(width * max(0.0, min(1.0, value / maximum)))
        if filled:
            pygame.draw.rect(self.screen, color, (x, y, filled, 7), border_radius=4)

    def draw_toast(self, message: str) -> None:
        width = min(500, self.font_sm.size(message)[0] + 40)
        rect = pygame.Rect((MAP_RIGHT - width) // 2, 618, width, 38)
        pygame.draw.rect(self.screen, (30, 45, 56), rect, border_radius=9)
        pygame.draw.rect(self.screen, CYAN, rect, 1, border_radius=9)
        self.text(message, rect.center, self.font_sm, INK, center=True)

    def draw_help(self) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((2, 6, 12, 220))
        self.screen.blit(shade, (0, 0))
        rect = pygame.Rect(245, 96, 790, 610)
        pygame.draw.rect(self.screen, (14, 27, 39), rect, border_radius=18)
        pygame.draw.rect(self.screen, CYAN, rect, 1, border_radius=18)
        self.text("CANCER KINGDOM", (640, 133), self.font_title, INK, center=True)
        self.text("Destroy the evolving ecosystem — preserve the human host", (640, 180), self.font_md, CYAN, center=True)

        items = [
            ("1", "SELECT", "Click an organ to inspect its health, tumor burden, and resistance."),
            ("2", "TREAT", "Spend 3 action points each turn. Every therapy has a biological trade-off."),
            ("3", "EVOLVE", "End the turn. Cancer grows, mutates under pressure, and can metastasize."),
            ("4", "SURVIVE", "Eliminate cancer or hold it below 8 burden through turn 35 without organ failure."),
        ]
        for index, (number, title, copy) in enumerate(items):
            y = 242 + index * 82
            pygame.draw.circle(self.screen, (25, 75, 79), (300, y + 14), 22)
            self.text(number, (300, y + 14), self.font_md, CYAN, center=True)
            self.text(title, (340, y), self.font_sm, YELLOW)
            self.draw_wrapped(copy, pygame.Rect(340, y + 24, 625, 45), self.font_sm, INK, 20)

        self.text("Sensitive cells", (360, 578), self.font_xs, RED)
        self.text("Resistant clones", (535, 578), self.font_xs, MAGENTA)
        self.text("Healthy tissue", (730, 578), self.font_xs, GREEN)
        pygame.draw.circle(self.screen, RED, (343, 585), 5)
        pygame.draw.circle(self.screen, MAGENTA, (518, 585), 5)
        pygame.draw.circle(self.screen, GREEN, (713, 585), 5)

        pygame.draw.rect(self.screen, (27, 99, 99), (440, 629, 400, 48), border_radius=10)
        self.text("CLICK ANYWHERE TO BEGIN", (640, 653), self.font_md, INK, center=True)
        self.text("Keys: 1–6 treatments  •  Space end turn  •  R restart  •  H help", (640, 690), self.font_xs, MUTED, center=True)

    def draw_game_over(self) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((1, 5, 10, 210))
        self.screen.blit(shade, (0, 0))
        won = self.sim.status == "won"
        rect = pygame.Rect(345, 205, 590, 370)
        pygame.draw.rect(self.screen, (15, 27, 39), rect, border_radius=18)
        pygame.draw.rect(self.screen, GREEN if won else RED, rect, 2, border_radius=18)
        self.text("REMISSION" if won else "HOST FAILURE", (640, 256), self.font_huge, GREEN if won else RED, center=True)
        outcome = "The cancer kingdom has fallen." if won else "Cancer has overwhelmed the body's defenses."
        self.text(outcome, (640, 321), self.font_md, INK, center=True)
        self.text(f"Turns survived  {self.sim.turn - 1}", (640, 370), self.font_sm, MUTED, center=True)
        self.text(f"Final burden  {self.sim.total_cancer:.1f}", (640, 397), self.font_sm, MUTED, center=True)
        self.text(f"Body health  {self.sim.body_health:.0f}%", (640, 424), self.font_sm, MUTED, center=True)
        button = pygame.Rect(490, 500, 300, 56)
        hovered = button.collidepoint(self.mouse)
        pygame.draw.rect(self.screen, (35, 128, 119) if hovered else (26, 89, 87), button, border_radius=10)
        self.text("NEW PATIENT", button.center, self.font_md, INK, center=True)

    def draw_wrapped(
        self, text: str, rect: pygame.Rect, font: pygame.font.Font, color: Tuple[int, int, int], line_height: int
    ) -> None:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= rect.width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        for index, line in enumerate(lines):
            if rect.y + index * line_height + line_height > rect.bottom:
                break
            self.text(line, (rect.x, rect.y + index * line_height), font, color)

    def text(
        self,
        value: str,
        position: Tuple[int, int],
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        *,
        center: bool = False,
        right: bool = False,
    ) -> None:
        surface = font.render(value, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = position
        elif right:
            rect.topright = position
        else:
            rect.topleft = position
        self.screen.blit(surface, rect)


def mix(a: Tuple[int, int, int], b: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(x + (y - x) * amount) for x, y in zip(a, b))


def shorten(text: str, length: int) -> str:
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def main() -> None:
    game = CancerKingdomGame()
    game.run()


if __name__ == "__main__":
    main()
