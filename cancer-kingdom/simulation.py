"""Core turn-based simulation for Cancer Kingdom.

The simulation deliberately has no pygame dependency.  It can be driven by the
human UI today and wrapped in a Gymnasium environment later.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple


@dataclass
class Organ:
    name: str
    health: float
    sensitive: float
    resistant: float
    vascularity: float
    immune: float
    niche: float
    vital_weight: float

    @property
    def cancer(self) -> float:
        return self.sensitive + self.resistant

    @property
    def resistance(self) -> float:
        if self.cancer <= 0.001:
            return 0.0
        return 100.0 * self.resistant / self.cancer


@dataclass(frozen=True)
class Treatment:
    key: str
    name: str
    cost: int
    targeted: bool
    description: str


TREATMENTS: Dict[str, Treatment] = {
    "chemo": Treatment(
        "chemo", "Chemotherapy", 2, False,
        "Systemic attack. Strong against sensitive cells; toxic to healthy tissue.",
    ),
    "radiation": Treatment(
        "radiation", "Radiation", 2, True,
        "Powerful local damage with moderate organ toxicity.",
    ),
    "surgery": Treatment(
        "surgery", "Surgery", 3, True,
        "Remove most of a visible tumor, but not hidden colonies.",
    ),
    "anti_angio": Treatment(
        "anti_angio", "Anti-angiogenesis", 1, True,
        "Cut tumor blood supply. Slows growth but increases hypoxic pressure.",
    ),
    "targeted": Treatment(
        "targeted", "Targeted therapy", 2, True,
        "Devastating to sensitive clones; resistant clones are spared.",
    ),
    "immuno": Treatment(
        "immuno", "Immunotherapy", 2, False,
        "Raises immune activity throughout the body for several turns.",
    ),
}


class CancerKingdom:
    """A compact population-level model of treatment and tumor evolution."""

    MAX_TURNS = 35
    ACTION_POINTS = 3

    ORGAN_TEMPLATE = (
        ("Brain", 1.00, 1.45),
        ("Lungs", 1.18, 1.25),
        ("Lymph Nodes", 1.12, 0.75),
        ("Liver", 1.25, 1.30),
        ("Stomach", 0.96, 0.75),
        ("Kidneys", 1.02, 1.00),
        ("Bone Marrow", 1.08, 1.20),
        ("Bone", 0.88, 0.75),
    )

    ROUTES = {
        "Brain": ("Lungs", "Bone"),
        "Lungs": ("Brain", "Lymph Nodes", "Liver"),
        "Lymph Nodes": ("Lungs", "Liver", "Stomach"),
        "Liver": ("Lungs", "Lymph Nodes", "Stomach", "Kidneys", "Bone Marrow"),
        "Stomach": ("Lymph Nodes", "Liver", "Kidneys"),
        "Kidneys": ("Liver", "Stomach", "Bone Marrow"),
        "Bone Marrow": ("Liver", "Kidneys", "Bone"),
        "Bone": ("Brain", "Bone Marrow"),
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.organs: Dict[str, Organ] = {}
        self.turn = 1
        self.action_points = self.ACTION_POINTS
        self.immune_boost_turns = 0
        self.treatment_pressure: Dict[str, float] = {}
        self.hypoxia: Dict[str, float] = {}
        self.log: List[str] = []
        self.status = "playing"
        self.clear_turns = 0
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng.seed(self.seed)
        self.organs = {
            name: Organ(name, 100.0, 0.0, 0.0, 58.0, 52.0, niche, weight)
            for name, niche, weight in self.ORGAN_TEMPLATE
        }
        self.organs["Lungs"].sensitive = 8.0
        self.organs["Lungs"].resistant = 0.25
        self.organs["Lungs"].vascularity = 68.0
        self.organs["Lungs"].immune = 43.0
        self.turn = 1
        self.action_points = self.ACTION_POINTS
        self.immune_boost_turns = 0
        self.treatment_pressure = {name: 0.0 for name in self.organs}
        self.hypoxia = {name: 0.0 for name in self.organs}
        self.status = "playing"
        self.clear_turns = 0
        self.log = [
            "A mutated colony has been detected in the lungs.",
            "Select an organ and balance tumor control against host damage.",
        ]

    @property
    def total_cancer(self) -> float:
        return sum(organ.cancer for organ in self.organs.values())

    @property
    def body_health(self) -> float:
        weight = sum(organ.vital_weight for organ in self.organs.values())
        return sum(organ.health * organ.vital_weight for organ in self.organs.values()) / weight

    @property
    def metastatic_organs(self) -> int:
        return sum(organ.cancer >= 0.4 for organ in self.organs.values())

    @property
    def progression(self) -> str:
        if self.total_cancer < 0.4:
            return "REMISSION"
        if self.metastatic_organs >= 4:
            return "LEVEL 6 — MULTI-ORGAN"
        if self.metastatic_organs >= 2:
            return "LEVEL 5 — METASTASIS"
        burden = max(organ.cancer for organ in self.organs.values())
        if burden >= 48:
            return "LEVEL 4 — VASCULAR INVASION"
        if burden >= 28:
            return "LEVEL 3 — TISSUE INVASION"
        if burden >= 12:
            return "LEVEL 2 — ESTABLISHED TUMOR"
        return "LEVEL 1 — LOCAL COLONY"

    def can_treat(self, treatment_key: str, target: Optional[str]) -> Tuple[bool, str]:
        if self.status != "playing":
            return False, "The campaign has ended."
        treatment = TREATMENTS[treatment_key]
        if treatment.cost > self.action_points:
            return False, "Not enough action points."
        if treatment.targeted and target not in self.organs:
            return False, "Select an organ first."
        if treatment.targeted and self.organs[target].cancer < 0.05:
            return False, "No detectable tumor in that organ."
        return True, ""

    def treat(self, treatment_key: str, target: Optional[str] = None) -> bool:
        allowed, reason = self.can_treat(treatment_key, target)
        if not allowed:
            self._add_log(reason)
            return False

        treatment = TREATMENTS[treatment_key]
        self.action_points -= treatment.cost

        if treatment_key == "chemo":
            killed = 0.0
            for organ in self.organs.values():
                killed += self._kill(organ, 0.36, 0.07)
                organ.health = max(0.0, organ.health - 4.2)
                organ.immune = max(5.0, organ.immune - 3.0)
                self.treatment_pressure[organ.name] += 18.0
            self._add_log(f"Chemotherapy removed {killed:.1f} tumor burden; all organs took toxicity.")

        elif treatment_key == "radiation":
            organ = self.organs[target]
            killed = self._kill(organ, 0.48, 0.18)
            organ.health = max(0.0, organ.health - 7.0)
            self.treatment_pressure[target] += 24.0
            self._add_log(f"Radiation struck {target}: {killed:.1f} tumor burden removed.")

        elif treatment_key == "surgery":
            organ = self.organs[target]
            killed = self._kill(organ, 0.72, 0.62)
            organ.health = max(0.0, organ.health - 12.0)
            organ.vascularity = max(20.0, organ.vascularity - 8.0)
            self.treatment_pressure[target] += 8.0
            self._add_log(f"Surgery removed {killed:.1f} tumor burden from {target}.")

        elif treatment_key == "anti_angio":
            organ = self.organs[target]
            old = organ.vascularity
            organ.vascularity = max(12.0, organ.vascularity - 24.0)
            killed = self._kill(organ, 0.10, 0.06)
            self.hypoxia[target] = min(100.0, self.hypoxia[target] + 28.0)
            self.treatment_pressure[target] += 10.0
            self._add_log(
                f"Blood supply in {target} fell {old - organ.vascularity:.0f}%; hypoxia is rising."
            )

        elif treatment_key == "targeted":
            organ = self.organs[target]
            killed = self._kill(organ, 0.58, 0.025)
            organ.health = max(0.0, organ.health - 2.0)
            self.treatment_pressure[target] += 34.0
            self._add_log(f"Targeted therapy removed {killed:.1f} burden from {target}.")

        elif treatment_key == "immuno":
            for organ in self.organs.values():
                organ.immune = min(100.0, organ.immune + 17.0)
                organ.health = max(0.0, organ.health - 1.2)
            self.immune_boost_turns = max(self.immune_boost_turns, 3)
            self._add_log("Immune activity rose throughout the body for three turns.")

        self._remove_trace_populations()
        self._check_terminal()
        return True

    def end_turn(self) -> None:
        if self.status != "playing":
            return

        events: List[str] = []
        for name, organ in self.organs.items():
            if organ.cancer < 0.05:
                self._recover_healthy_organ(organ)
                continue

            suppression = min(0.72, organ.cancer / 120.0)
            immune_power = organ.immune * (1.0 - suppression)
            if self.immune_boost_turns:
                immune_power *= 1.22
            immune_kill_rate = min(0.16, immune_power / 700.0)
            immune_killed = self._kill(organ, immune_kill_rate, immune_kill_rate * 0.62)

            carrying_room = max(0.04, 1.0 - organ.cancer / 115.0)
            blood_factor = 0.35 + 0.009 * organ.vascularity
            sensitive_growth = 0.18 * organ.niche * blood_factor * carrying_room
            resistant_growth = 0.145 * organ.niche * blood_factor * carrying_room
            organ.sensitive += organ.sensitive * sensitive_growth
            organ.resistant += organ.resistant * resistant_growth

            pressure = self.treatment_pressure[name]
            hypoxia = self.hypoxia[name]
            conversion_rate = min(0.09, 0.003 + pressure / 1450.0 + hypoxia / 2700.0)
            converted = organ.sensitive * conversion_rate
            organ.sensitive -= converted
            organ.resistant += converted

            organ.health = max(0.0, organ.health - (0.22 + organ.cancer * 0.082))
            organ.immune = max(6.0, organ.immune - organ.cancer * 0.035 + 0.8)
            organ.vascularity = min(100.0, organ.vascularity + organ.cancer * 0.10)
            self.treatment_pressure[name] *= 0.46
            self.hypoxia[name] *= 0.72

            if immune_killed >= 0.7:
                events.append(f"Immune cells cleared {immune_killed:.1f} burden in {name}.")

        spread_events = self._metastasize()
        events.extend(spread_events)

        self.immune_boost_turns = max(0, self.immune_boost_turns - 1)
        self.turn += 1
        self.action_points = self.ACTION_POINTS
        self._remove_trace_populations()

        if events:
            for event in events[-2:]:
                self._add_log(event)
        else:
            self._add_log("Cancer populations evolved; no new metastasis was detected.")
        self._check_terminal()

    def _metastasize(self) -> List[str]:
        events: List[str] = []
        sources = [organ for organ in self.organs.values() if organ.cancer >= 18.0]
        self.rng.shuffle(sources)
        for source in sources:
            chance = min(0.48, 0.025 + source.cancer / 190.0 + source.vascularity / 700.0)
            if self.rng.random() >= chance:
                continue
            candidates = list(self.ROUTES[source.name])
            weights = [self.organs[name].niche / (1.0 + self.organs[name].cancer) for name in candidates]
            destination_name = self.rng.choices(candidates, weights=weights, k=1)[0]
            destination = self.organs[destination_name]
            if destination.cancer < 0.05:
                seed_size = 0.75 + self.rng.random() * 0.8
                resistant_share = min(0.8, source.resistant / max(0.01, source.cancer))
                destination.sensitive += seed_size * (1.0 - resistant_share)
                destination.resistant += seed_size * resistant_share
                events.append(f"Metastasis: {source.name} seeded a colony in {destination_name}!")
            else:
                migration = 0.25 + self.rng.random() * 0.5
                destination.sensitive += migration * 0.65
                destination.resistant += migration * 0.35
        return events

    def _recover_healthy_organ(self, organ: Organ) -> None:
        organ.health = min(100.0, organ.health + 0.65)
        organ.immune = min(62.0, organ.immune + 0.6)
        organ.vascularity += (58.0 - organ.vascularity) * 0.08
        self.treatment_pressure[organ.name] *= 0.45
        self.hypoxia[organ.name] *= 0.7

    @staticmethod
    def _kill(organ: Organ, sensitive_rate: float, resistant_rate: float) -> float:
        sensitive_killed = organ.sensitive * sensitive_rate
        resistant_killed = organ.resistant * resistant_rate
        organ.sensitive -= sensitive_killed
        organ.resistant -= resistant_killed
        return sensitive_killed + resistant_killed

    def _remove_trace_populations(self) -> None:
        for organ in self.organs.values():
            if organ.sensitive < 0.035:
                organ.sensitive = 0.0
            if organ.resistant < 0.025:
                organ.resistant = 0.0

    def _check_terminal(self) -> None:
        if self.total_cancer < 0.12:
            self.clear_turns += 1
        else:
            self.clear_turns = 0

        critical_failure = any(
            organ.health <= (10.0 if organ.vital_weight >= 1.0 else 5.0)
            for organ in self.organs.values()
        )
        if critical_failure or self.body_health <= 22.0 or self.total_cancer >= 430.0:
            self.status = "lost"
            self._add_log("Critical organ failure: the cancer kingdom has overwhelmed the host.")
        elif self.clear_turns >= 1:
            self.status = "won"
            self._add_log("No detectable cancer remains. The patient is in remission!")
        elif self.turn > self.MAX_TURNS:
            if self.total_cancer < 8.0 and self.body_health > 45.0:
                self.status = "won"
                self._add_log("The cancer is under durable control. The patient survives.")
            else:
                self.status = "lost"
                self._add_log("Treatment time is exhausted and disease remains uncontrolled.")

    def _add_log(self, message: str) -> None:
        self.log.append(message)
        self.log = self.log[-7:]

