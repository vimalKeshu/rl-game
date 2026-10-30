import unittest

from simulation import CancerKingdom


class CancerKingdomTests(unittest.TestCase):
    def test_targeted_treatment_costs_points_and_selects_resistance(self) -> None:
        sim = CancerKingdom(seed=2)
        before = sim.organs["Lungs"]
        initial_burden = before.cancer
        initial_resistance = before.resistance

        self.assertTrue(sim.treat("targeted", "Lungs"))

        self.assertEqual(sim.action_points, 1)
        self.assertLess(before.cancer, initial_burden)
        self.assertGreater(before.resistance, initial_resistance)

    def test_end_turn_grows_an_untreated_colony(self) -> None:
        sim = CancerKingdom(seed=2)
        initial_burden = sim.total_cancer

        sim.end_turn()

        self.assertEqual(sim.turn, 2)
        self.assertEqual(sim.action_points, 3)
        self.assertGreater(sim.total_cancer, initial_burden)

    def test_chemotherapy_is_systemic_and_toxic(self) -> None:
        sim = CancerKingdom(seed=2)
        initial_health = sim.body_health

        self.assertTrue(sim.treat("chemo"))

        self.assertLess(sim.body_health, initial_health)
        self.assertTrue(all(organ.health < 100 for organ in sim.organs.values()))

    def test_no_treatment_eventually_loses(self) -> None:
        sim = CancerKingdom(seed=3)
        while sim.status == "playing":
            sim.end_turn()

        self.assertEqual(sim.status, "lost")
        self.assertGreaterEqual(sim.metastatic_organs, 2)


if __name__ == "__main__":
    unittest.main()

