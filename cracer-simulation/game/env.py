from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple, Dict, Any


class Config:
    lane_count = 3
    road_width_ratio = 0.52
    road_min_width = 240
    road_max_width = 540
    road_side_margin = 80
    road_height_multiplier = 1.4
    enable_curves = False

    lane_marker_size = (8, 40)
    lane_marker_spacing = 80

    player_size = (34, 58)
    enemy_size = (34, 58)
    truck_size = (42, 120)
    race_car_size = (32, 54)
    fuel_size = (22, 28)
    pothole_size = (26, 26)
    bump_size = (36, 18)

    base_cruise_speed = 220
    base_max_speed = 360
    min_speed = 140
    accel = 180
    brake = 240
    turn_speed = 260
    traffic_base_ratio = 0.85

    fuel_drain_rate = 6
    fuel_pickup = 38
    crash_fuel_loss = 18
    pothole_fuel_loss = 6
    pothole_speed_penalty = 80
    bump_fuel_loss = 2
    bump_speed_penalty = 40
    max_fuel_pickups = 2
    max_potholes = 1
    max_bumps = 2

    enemy_speed_boost_per_stage = 12
    enemy_speed_min = -20
    enemy_speed_max = 110
    truck_speed_min = -10
    truck_speed_max = 80
    race_speed_min = 120
    race_speed_max = 220
    min_traffic_speed = 120
    truck_chance_base = 0.1
    truck_chance_per_stage = 0.05
    truck_chance_max = 0.35
    race_chance_base = 0.08
    race_chance_per_stage = 0.04
    race_chance_max = 0.28

    pothole_base_interval = 4.6
    pothole_interval_drop = 0.05
    pothole_min_interval = 3.2
    bump_base_interval = 5.2
    bump_interval_drop = 0.06
    bump_min_interval = 3.6

    speed_zone_base_interval = 7.0
    speed_zone_jitter = 2.0
    speed_zone_offset_range = (-50, 70)
    speed_limit_buffer = 80

    slope_base_interval = 6.5
    slope_jitter = 2.5
    slope_accel = 48
    slope_max_boost = 40

    stage_distance = 4200
    starting_lives = 3


@dataclass
class Entity:
    kind: str
    x: float
    y: float
    width: float
    height: float
    speed_offset: float = 0.0


class CracerEnv:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        fps: int = 60,
        render_mode: Optional[str] = None,
        obs_mode: str = "state",
        action_mode: str = "discrete",
        seed: Optional[int] = None,
        max_objects: int = 6,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / float(fps)
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.max_objects = max_objects
        self.object_types = ["enemy", "truck", "race", "fuel", "pothole", "bump"]

        self.seed_value = seed
        self.rng = random.Random(seed)

        self.renderer = None

        self.road_width = 0.0
        self.road_height = 0.0
        self.road_offset = 0.0
        self.target_road_offset = 0.0
        self.curve_timer = 0.0

        self.lane_markers: List[List[float]] = []
        self.enemies: List[Entity] = []
        self.pickups: List[Entity] = []
        self.hazards: List[Entity] = []

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.stage = 1
        self.score = 0.0
        self.fuel = 100.0
        self.max_fuel = 100.0
        self.lives = Config.starting_lives
        self.distance_remaining = float(Config.stage_distance)

        self.cruise_speed = Config.base_cruise_speed
        self.max_speed = Config.base_max_speed
        self.current_speed = Config.base_cruise_speed
        self.speed_zone_offset = 0.0
        self.speed_limit = Config.base_cruise_speed
        self.slope_direction = 0.0

        self.game_mode = "playing"
        self.invincible_time = 0.0
        self.bump_cooldown = 0.0
        self.stage_clear_time = 0.0

        self.enemy_spawn_timer = 0.0
        self.fuel_spawn_timer = 0.0
        self.pothole_spawn_timer = 0.0
        self.bump_spawn_timer = 0.0
        self.speed_zone_timer = 0.0
        self.slope_timer = 0.0
        self.message_timer = 0.0
        self.message_text: Optional[str] = None

        self.player_x = 0.0
        self.player_y = 0.0
        self.player_width = float(Config.player_size[0])
        self.player_height = float(Config.player_size[1])

        self.layout_scene()
        self.reset()

    def seed(self, seed: Optional[int]) -> None:
        self.seed_value = seed
        self.rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        self.enemies = []
        self.pickups = []
        self.hazards = []
        self.lane_markers = []

        self.stage = 1
        self.score = 0.0
        self.fuel = self.max_fuel
        self.lives = Config.starting_lives
        self.distance_remaining = float(Config.stage_distance)

        self.cruise_speed = Config.base_cruise_speed
        self.max_speed = Config.base_max_speed
        self.current_speed = self.cruise_speed
        self.speed_zone_offset = 0.0
        self.speed_zone_timer = 0.0
        self.speed_limit = self.cruise_speed
        self.slope_timer = 0.0
        self.slope_direction = 0.0

        self.invincible_time = 0.0
        self.bump_cooldown = 0.0
        self.stage_clear_time = 0.0
        self.game_mode = "playing"
        self.message_timer = 0.0
        self.message_text = None

        self.road_offset = 0.0
        self.target_road_offset = 0.0
        self.curve_timer = 0.0

        self.enemy_spawn_timer = 0.0
        self.fuel_spawn_timer = 0.0
        self.pothole_spawn_timer = 0.0
        self.bump_spawn_timer = 0.0

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.player_x = self.width / 2.0
        self.player_y = self.height * 0.2

        self.layout_scene()

        return self._get_observation(), self._get_info()

    def layout_scene(self) -> None:
        max_width = max(100.0, self.width - Config.road_side_margin)
        proposed_width = self.width * Config.road_width_ratio
        self.road_width = min(
            max(Config.road_min_width, proposed_width),
            min(Config.road_max_width, max_width),
        )
        self.road_height = self.height * Config.road_height_multiplier
        self.rebuild_lane_markers()
        self.clamp_player_to_road()

    def rebuild_lane_markers(self) -> None:
        self.lane_markers = []
        if Config.lane_count <= 1:
            return
        lane_width = self.current_lane_width
        marker_count = int(self.road_height / Config.lane_marker_spacing) + 3
        for lane in range(1, Config.lane_count):
            x = -self.road_width / 2.0 + lane_width * lane
            for index in range(marker_count):
                y = -self.road_height / 2.0 + index * Config.lane_marker_spacing
                self.lane_markers.append([x, y])

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        prev_score = self.score
        self.apply_action(action)

        dt = self.dt
        self.update_road_curve(dt)
        self.update_timers(dt)
        self.update_speed_zones(dt)
        self.update_slope(dt)
        self.update_speed(dt)
        self.update_player(dt)

        if self.game_mode != "game_over":
            self.move_lane_markers(dt)
            self.move_entities(dt)

        if self.game_mode in ("playing", "crashed"):
            self.update_spawns(dt)
            self.update_fuel_and_distance(dt)

        self.handle_collisions()

        reward = self.score - prev_score
        terminated = self.game_mode == "game_over"
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self, mode: Optional[str] = None) -> Optional["Any"]:
        if mode is None:
            mode = self.render_mode
        if mode is None:
            return None

        if self.renderer is None or self.renderer.mode != mode:
            from .render import PygameRenderer

            if self.renderer is not None:
                self.renderer.close()
            self.renderer = PygameRenderer(self.width, self.height, mode)
        frame = self.renderer.draw(self)
        if mode == "human":
            self.renderer.tick(self.fps)
            return None
        return frame

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def apply_action(self, action: Any) -> None:
        left = right = up = down = False
        if self.action_mode == "discrete":
            try:
                action_id = int(action)
            except (TypeError, ValueError):
                action_id = 0
            if action_id == 1:
                left = True
            elif action_id == 2:
                right = True
            elif action_id == 3:
                up = True
            elif action_id == 4:
                down = True
            elif action_id == 5:
                left = True
                up = True
            elif action_id == 6:
                right = True
                up = True
            elif action_id == 7:
                left = True
                down = True
            elif action_id == 8:
                right = True
                down = True
        elif self.action_mode == "continuous":
            try:
                steer, throttle = action
            except (TypeError, ValueError):
                steer, throttle = 0.0, 0.0
            deadzone = 0.1
            left = steer < -deadzone
            right = steer > deadzone
            up = throttle > deadzone
            down = throttle < -deadzone
        elif self.action_mode == "buttons":
            if isinstance(action, dict):
                left = bool(action.get("left", False))
                right = bool(action.get("right", False))
                up = bool(action.get("accelerate", False))
                down = bool(action.get("brake", False))
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        if left and right:
            left = right = False
        if up and down:
            up = down = False

        self.left_pressed = left
        self.right_pressed = right
        self.up_pressed = up
        self.down_pressed = down

    @staticmethod
    def action_from_buttons(left: bool, right: bool, up: bool, down: bool) -> int:
        if left and right:
            left = right = False
        if up and down:
            up = down = False
        if left and up:
            return 5
        if right and up:
            return 6
        if left and down:
            return 7
        if right and down:
            return 8
        if left:
            return 1
        if right:
            return 2
        if up:
            return 3
        if down:
            return 4
        return 0

    @property
    def current_lane_width(self) -> float:
        return self.road_width / float(Config.lane_count)

    @property
    def traffic_base_speed(self) -> float:
        return max(self.current_speed, self.speed_limit * Config.traffic_base_ratio)

    def lane_center_x(self, lane_index: int) -> float:
        road_center_x = self.width / 2.0 + self.road_offset
        return road_center_x - self.road_width / 2.0 + self.current_lane_width * (lane_index + 0.5)

    def update_road_curve(self, dt: float) -> None:
        if not Config.enable_curves:
            if self.road_offset != 0.0:
                delta = -self.road_offset
                self.road_offset = 0.0
                self.target_road_offset = 0.0
                self.shift_entities(delta)
            return

        self.curve_timer += dt
        if self.curve_timer > 2.2:
            self.curve_timer = 0.0
            max_offset = min(self.road_width * 0.22, self.width * 0.2)
            self.target_road_offset = self.rng.uniform(-max_offset, max_offset)

        previous_offset = self.road_offset
        damping = 2.0
        self.road_offset += (self.target_road_offset - self.road_offset) * dt * damping
        delta = self.road_offset - previous_offset
        if abs(delta) > 0.001:
            self.shift_entities(delta)

    def shift_entities(self, delta: float) -> None:
        for entity in self.enemies:
            entity.x += delta
        for entity in self.pickups:
            entity.x += delta
        for entity in self.hazards:
            entity.x += delta

    def update_timers(self, dt: float) -> None:
        if self.invincible_time > 0.0:
            self.invincible_time -= dt
            if self.invincible_time <= 0.0:
                self.invincible_time = 0.0
                if self.game_mode == "crashed":
                    self.game_mode = "playing"

        if self.stage_clear_time > 0.0:
            self.stage_clear_time -= dt
            if self.stage_clear_time <= 0.0:
                self.stage_clear_time = 0.0
                self.advance_stage()

        if self.message_timer > 0.0:
            self.message_timer -= dt
            if self.message_timer <= 0.0:
                self.message_timer = 0.0
                self.message_text = None

        if self.bump_cooldown > 0.0:
            self.bump_cooldown -= dt
            if self.bump_cooldown < 0.0:
                self.bump_cooldown = 0.0

    def update_speed_zones(self, dt: float) -> None:
        if self.game_mode == "game_over":
            return
        self.speed_zone_timer -= dt
        if self.speed_zone_timer <= 0.0:
            jitter = self.rng.uniform(-Config.speed_zone_jitter, Config.speed_zone_jitter)
            self.speed_zone_timer = max(2.5, Config.speed_zone_base_interval + jitter)
            self.speed_zone_offset = self.rng.uniform(*Config.speed_zone_offset_range)

        min_limit = Config.min_speed + 20
        max_limit = self.max_speed - 20
        suggested_limit = self.cruise_speed + self.speed_zone_offset
        self.speed_limit = min(max_limit, max(min_limit, suggested_limit))

    def update_slope(self, dt: float) -> None:
        if self.game_mode == "game_over":
            return
        self.slope_timer -= dt
        if self.slope_timer <= 0.0:
            jitter = self.rng.uniform(-Config.slope_jitter, Config.slope_jitter)
            self.slope_timer = max(3.0, Config.slope_base_interval + jitter)
            roll = self.rng.uniform(0.0, 1.0)
            if roll < 0.33:
                self.slope_direction = -1.0
            elif roll < 0.66:
                self.slope_direction = 1.0
            else:
                self.slope_direction = 0.0

    def update_speed(self, dt: float) -> None:
        if self.game_mode == "game_over":
            self.current_speed = 0.0
            return

        max_allowed = self.max_speed + (Config.slope_max_boost if self.slope_direction > 0 else 0)
        if self.up_pressed:
            self.current_speed = min(max_allowed, self.current_speed + Config.accel * dt)
        elif self.down_pressed:
            self.current_speed = max(Config.min_speed, self.current_speed - Config.brake * dt)
        else:
            adjustment = (self.speed_limit - self.current_speed) * min(1.0, dt * 2.0)
            self.current_speed += adjustment

        if self.slope_direction != 0.0:
            self.current_speed += self.slope_direction * Config.slope_accel * dt

        self.current_speed = min(max(self.current_speed, Config.min_speed), max_allowed)

    def update_player(self, dt: float) -> None:
        if self.left_pressed:
            self.player_x -= Config.turn_speed * dt
        if self.right_pressed:
            self.player_x += Config.turn_speed * dt
        self.clamp_player_to_road()

    def clamp_player_to_road(self) -> None:
        road_center_x = self.width / 2.0 + self.road_offset
        min_x = road_center_x - self.road_width / 2.0 + self.player_width / 2.0 + 4
        max_x = road_center_x + self.road_width / 2.0 - self.player_width / 2.0 - 4
        self.player_x = min(max(self.player_x, min_x), max_x)
        self.player_y = self.height * 0.2

    def move_lane_markers(self, dt: float) -> None:
        delta = self.current_speed * dt
        wrap_y = self.road_height / 2.0 + Config.lane_marker_spacing
        for marker in self.lane_markers:
            marker[1] -= delta
            if marker[1] < -wrap_y:
                marker[1] += self.road_height + Config.lane_marker_spacing

    def move_entities(self, dt: float) -> None:
        traffic_speed = self.traffic_base_speed
        enemy_min_y = -self.height * 0.2

        for entity in list(self.enemies):
            entity.y -= (traffic_speed + entity.speed_offset) * dt
            if entity.y < enemy_min_y:
                self.enemies.remove(entity)

        for entity in list(self.pickups):
            entity.y -= (self.current_speed + entity.speed_offset) * dt
            if entity.y < enemy_min_y:
                self.pickups.remove(entity)

        for entity in list(self.hazards):
            entity.y -= (self.current_speed + entity.speed_offset) * dt
            if entity.y < enemy_min_y:
                self.hazards.remove(entity)

    def update_spawns(self, dt: float) -> None:
        self.enemy_spawn_timer += dt
        self.fuel_spawn_timer += dt
        self.pothole_spawn_timer += dt
        self.bump_spawn_timer += dt

        base_interval = max(0.6, 1.2 - float(self.stage) * 0.05)
        if self.enemy_spawn_timer > base_interval:
            self.enemy_spawn_timer = 0.0
            self.spawn_enemy()

        fuel_interval = max(2.5, 4.2 - float(self.stage) * 0.15)
        if self.fuel_spawn_timer > fuel_interval:
            self.fuel_spawn_timer = 0.0
            if len(self.pickups) < Config.max_fuel_pickups:
                self.spawn_fuel()

        pothole_interval = max(
            Config.pothole_min_interval,
            Config.pothole_base_interval - float(self.stage) * Config.pothole_interval_drop,
        )
        if self.pothole_spawn_timer > pothole_interval:
            self.pothole_spawn_timer = 0.0
            pothole_count = sum(1 for hazard in self.hazards if hazard.kind == "pothole")
            if pothole_count < Config.max_potholes:
                self.spawn_pothole()

        bump_interval = max(
            Config.bump_min_interval,
            Config.bump_base_interval - float(self.stage) * Config.bump_interval_drop,
        )
        if self.bump_spawn_timer > bump_interval:
            self.bump_spawn_timer = 0.0
            bump_count = sum(1 for hazard in self.hazards if hazard.kind == "bump")
            if bump_count < Config.max_bumps:
                self.spawn_speed_bump()

    def update_fuel_and_distance(self, dt: float) -> None:
        self.fuel -= Config.fuel_drain_rate * dt
        if self.fuel <= 0.0:
            self.fuel = 0.0
            self.trigger_game_over()
            return

        self.distance_remaining -= self.current_speed * dt
        self.score += self.current_speed * dt * 0.6

        if self.distance_remaining <= 0.0:
            self.distance_remaining = 0.0
            self.start_stage_clear()

    def is_spawn_clear(
        self, x: float, y: float, size: Tuple[float, float], avoiding: List[Entity]
    ) -> bool:
        padding_w = 8.0
        padding_h = 12.0
        spawn_w = size[0] + padding_w * 2.0
        spawn_h = size[1] + padding_h * 2.0
        for entity in avoiding:
            if self.rects_intersect(
                x,
                y,
                spawn_w,
                spawn_h,
                entity.x,
                entity.y,
                entity.width,
                entity.height,
            ):
                return False
        return True

    def find_spawn_x(
        self,
        size: Tuple[float, float],
        y: float,
        avoiding: List[Entity],
        jitter: float,
    ) -> Optional[float]:
        lanes = list(range(Config.lane_count))
        self.rng.shuffle(lanes)
        for lane_index in lanes:
            base_x = self.lane_center_x(lane_index)
            for _ in range(2):
                x = base_x + self.rng.uniform(-jitter, jitter)
                if self.is_spawn_clear(x, y, size, avoiding):
                    return x
        return None

    def spawn_enemy(self) -> None:
        lane_index = self.rng.randrange(Config.lane_count)
        lane_x = self.lane_center_x(lane_index)

        truck_chance = min(
            Config.truck_chance_max,
            Config.truck_chance_base + max(0, self.stage - 1) * Config.truck_chance_per_stage,
        )
        race_chance = min(
            Config.race_chance_max,
            Config.race_chance_base + max(0, self.stage - 1) * Config.race_chance_per_stage,
        )
        roll = self.rng.uniform(0.0, 1.0)
        is_truck = self.stage >= 2 and roll < truck_chance
        is_race = (not is_truck) and self.stage >= 1 and roll < truck_chance + race_chance

        if is_truck:
            size = Config.truck_size
            kind = "truck"
            jitter = self.current_lane_width * 0.08
        elif is_race:
            size = Config.race_car_size
            kind = "race"
            jitter = self.current_lane_width * 0.18
        else:
            size = Config.enemy_size
            kind = "enemy"
            jitter = self.current_lane_width * 0.18

        x = lane_x + self.rng.uniform(-jitter, jitter)
        y = self.height + size[1]

        stage_boost = float(self.stage - 1) * Config.enemy_speed_boost_per_stage
        speed_min = Config.race_speed_min if is_race else (Config.truck_speed_min if is_truck else Config.enemy_speed_min)
        speed_max = Config.race_speed_max if is_race else (Config.truck_speed_max if is_truck else Config.enemy_speed_max)
        speed_offset = stage_boost + self.rng.uniform(speed_min, speed_max)
        min_traffic = Config.min_traffic_speed + (80 if is_race else 0)
        minimum_offset = -self.traffic_base_speed + min_traffic
        if speed_offset < minimum_offset:
            speed_offset = minimum_offset

        self.enemies.append(Entity(kind=kind, x=x, y=y, width=size[0], height=size[1], speed_offset=speed_offset))

    def spawn_fuel(self) -> None:
        y = self.height + Config.fuel_size[1]
        jitter = self.current_lane_width * 0.12
        avoid = list(self.enemies)
        x = self.find_spawn_x(Config.fuel_size, y, avoid, jitter)
        if x is None:
            return
        speed_offset = self.rng.uniform(-20, 40)
        self.pickups.append(
            Entity(kind="fuel", x=x, y=y, width=Config.fuel_size[0], height=Config.fuel_size[1], speed_offset=speed_offset)
        )

    def spawn_pothole(self) -> None:
        y = self.height + Config.pothole_size[1]
        jitter = self.current_lane_width * 0.18
        avoid = list(self.enemies) + list(self.pickups) + list(self.hazards)
        x = self.find_spawn_x(Config.pothole_size, y, avoid, jitter)
        if x is None:
            return
        self.hazards.append(
            Entity(kind="pothole", x=x, y=y, width=Config.pothole_size[0], height=Config.pothole_size[1])
        )

    def spawn_speed_bump(self) -> None:
        y = self.height + Config.bump_size[1]
        jitter = self.current_lane_width * 0.15
        avoid = list(self.enemies) + list(self.pickups) + list(self.hazards)
        x = self.find_spawn_x(Config.bump_size, y, avoid, jitter)
        if x is None:
            return
        self.hazards.append(
            Entity(kind="bump", x=x, y=y, width=Config.bump_size[0], height=Config.bump_size[1])
        )

    def handle_collisions(self) -> None:
        for enemy in list(self.enemies):
            if self.intersects_player(enemy):
                self.handle_enemy_collision(enemy)

        for fuel in list(self.pickups):
            if self.intersects_player(fuel):
                self.handle_fuel_pickup(fuel)

        for hazard in list(self.hazards):
            if self.intersects_player(hazard):
                if hazard.kind == "pothole":
                    self.handle_pothole_hit(hazard)
                elif hazard.kind == "bump":
                    self.handle_speed_bump_hit(hazard)

    def handle_enemy_collision(self, enemy: Entity) -> None:
        if self.invincible_time > 0.0 or self.game_mode not in ("playing", "crashed"):
            return
        self.invincible_time = 1.0
        self.message_text = "CRASH!"
        self.message_timer = 0.8
        self.game_mode = "crashed"
        self.fuel = max(0.0, self.fuel - Config.crash_fuel_loss)
        self.lives = max(0, self.lives - 1)
        self.current_speed = max(Config.min_speed, self.current_speed * 0.55)
        if enemy in self.enemies:
            self.enemies.remove(enemy)
        if self.fuel <= 0.0 or self.lives <= 0:
            self.trigger_game_over()

    def handle_fuel_pickup(self, fuel: Entity) -> None:
        self.fuel = min(self.max_fuel, self.fuel + Config.fuel_pickup)
        self.score += 250
        if fuel in self.pickups:
            self.pickups.remove(fuel)

    def handle_pothole_hit(self, pothole: Entity) -> None:
        if self.bump_cooldown > 0.0 or self.game_mode not in ("playing", "crashed"):
            return
        self.bump_cooldown = 0.6
        self.message_text = "POTHOLE!"
        self.message_timer = 0.5
        self.fuel = max(0.0, self.fuel - Config.pothole_fuel_loss)
        self.current_speed = max(Config.min_speed, self.current_speed - Config.pothole_speed_penalty)
        if pothole in self.hazards:
            self.hazards.remove(pothole)
        if self.fuel <= 0.0:
            self.trigger_game_over()

    def handle_speed_bump_hit(self, bump: Entity) -> None:
        if self.bump_cooldown > 0.0 or self.game_mode not in ("playing", "crashed"):
            return
        self.bump_cooldown = 0.4
        self.message_text = "BUMP!"
        self.message_timer = 0.4
        self.fuel = max(0.0, self.fuel - Config.bump_fuel_loss)
        self.current_speed = max(Config.min_speed, self.current_speed - Config.bump_speed_penalty)
        if bump in self.hazards:
            self.hazards.remove(bump)
        if self.fuel <= 0.0:
            self.trigger_game_over()

    def start_stage_clear(self) -> None:
        if self.game_mode == "stage_clear":
            return
        self.game_mode = "stage_clear"
        self.stage_clear_time = 2.0
        self.score += float(1000 * self.stage)

    def advance_stage(self) -> None:
        self.stage += 1
        self.distance_remaining = float(Config.stage_distance + (self.stage - 1) * 500)
        self.fuel = min(self.max_fuel, self.fuel + 25)
        self.cruise_speed += 18
        self.max_speed += 22
        self.speed_zone_offset = 0.0
        self.speed_zone_timer = 0.0
        self.speed_limit = self.cruise_speed
        self.slope_timer = 0.0
        self.slope_direction = 0.0
        self.game_mode = "playing"

    def trigger_game_over(self) -> None:
        if self.game_mode == "game_over":
            return
        self.game_mode = "game_over"
        self.current_speed = 0.0

    def intersects_player(self, entity: Entity) -> bool:
        return self.rects_intersect(
            self.player_x,
            self.player_y,
            self.player_width,
            self.player_height,
            entity.x,
            entity.y,
            entity.width,
            entity.height,
        )

    @staticmethod
    def rects_intersect(
        ax: float,
        ay: float,
        aw: float,
        ah: float,
        bx: float,
        by: float,
        bw: float,
        bh: float,
    ) -> bool:
        return abs(ax - bx) * 2.0 < (aw + bw) and abs(ay - by) * 2.0 < (ah + bh)

    def _get_observation(self) -> Any:
        if self.obs_mode == "state":
            return self._get_state_observation()
        if self.obs_mode in ("pixels", "rgb_array"):
            return self.render(mode="rgb_array")
        raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

    def _get_state_observation(self) -> List[float]:
        road_center_x = self.width / 2.0 + self.road_offset
        player_x_norm = (self.player_x - road_center_x) / (self.road_width / 2.0)
        player_x_norm = max(-1.0, min(1.0, player_x_norm))

        max_allowed = self.max_speed + (Config.slope_max_boost if self.slope_direction > 0 else 0)
        speed_norm = (self.current_speed - Config.min_speed) / max(1.0, (max_allowed - Config.min_speed))
        speed_norm = max(0.0, min(1.0, speed_norm))

        speed_limit_norm = (self.speed_limit - Config.min_speed) / max(1.0, (self.max_speed - Config.min_speed))
        speed_limit_norm = max(0.0, min(1.0, speed_limit_norm))

        fuel_norm = max(0.0, min(1.0, self.fuel / self.max_fuel))
        distance_norm = max(0.0, min(1.0, self.distance_remaining / self._current_stage_distance()))
        stage_norm = max(0.0, min(1.0, self.stage / 10.0))
        lives_norm = max(0.0, min(1.0, self.lives / float(Config.starting_lives)))

        crashed_flag = 1.0 if self.game_mode == "crashed" else 0.0
        stage_clear_flag = 1.0 if self.game_mode == "stage_clear" else 0.0
        game_over_flag = 1.0 if self.game_mode == "game_over" else 0.0

        base = [
            player_x_norm,
            speed_norm,
            speed_limit_norm,
            fuel_norm,
            distance_norm,
            stage_norm,
            lives_norm,
            float(self.slope_direction),
            crashed_flag,
            stage_clear_flag,
            game_over_flag,
        ]

        objects = self._nearest_objects()
        for entity in objects:
            dx = (entity.x - self.player_x) / (self.road_width / 2.0)
            dy = (entity.y - self.player_y) / float(self.height)
            dx = max(-1.0, min(1.0, dx))
            dy = max(-1.0, min(1.0, dy))
            one_hot = [1.0 if entity.kind == kind else 0.0 for kind in self.object_types]
            base.extend([dx, dy] + one_hot)

        while len(base) < self.state_size:
            base.append(0.0)
        return base

    def _nearest_objects(self) -> List[Entity]:
        all_objects = self.enemies + self.pickups + self.hazards
        candidates = [obj for obj in all_objects if obj.y >= self.player_y - self.height * 0.1]
        candidates.sort(key=lambda obj: obj.y)
        return candidates[: self.max_objects]

    def _current_stage_distance(self) -> float:
        return float(Config.stage_distance + (self.stage - 1) * 500)

    @property
    def state_size(self) -> int:
        base_count = 11
        per_object = 2 + len(self.object_types)
        return base_count + self.max_objects * per_object

    def _get_info(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "speed": self.current_speed,
            "speed_limit": self.speed_limit,
            "fuel": self.fuel,
            "lives": self.lives,
            "stage": self.stage,
            "distance_remaining": self.distance_remaining,
            "game_mode": self.game_mode,
            "message": self.current_message(),
        }

    def current_message(self) -> Optional[str]:
        if self.game_mode == "game_over":
            return "GAME OVER"
        if self.game_mode == "stage_clear":
            return f"STAGE {self.stage} CLEAR"
        if self.message_timer > 0.0 and self.message_text:
            return self.message_text
        return None
