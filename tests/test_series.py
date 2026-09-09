"""Tests for the independent, constant-probability series approximation."""

import math
import unittest

from src.series import series_win_probability


class SeriesProbabilityTests(unittest.TestCase):
    def test_known_bo3_value(self) -> None:
        self.assertAlmostEqual(series_win_probability(0.7, 3), 0.784)

    def test_known_bo5_value(self) -> None:
        self.assertAlmostEqual(series_win_probability(0.7, 5), 0.83692)

    def test_bo1_is_identity(self) -> None:
        for p in (0.0, 0.1, 0.5, 0.7, 1.0):
            with self.subTest(p=p):
                self.assertAlmostEqual(series_win_probability(p, 1), p)

    def test_endpoints_and_fair_series(self) -> None:
        for best_of in (1, 3, 5):
            with self.subTest(best_of=best_of):
                self.assertEqual(series_win_probability(0.0, best_of), 0.0)
                self.assertEqual(series_win_probability(1.0, best_of), 1.0)
                self.assertEqual(series_win_probability(0.5, best_of), 0.5)

    def test_swapping_teams_gives_complement(self) -> None:
        for best_of in (1, 3, 5):
            for p in (0.01, 0.1, 0.23, 0.7, 0.99):
                with self.subTest(best_of=best_of, p=p):
                    self.assertAlmostEqual(
                        series_win_probability(p, best_of)
                        + series_win_probability(1.0 - p, best_of),
                        1.0,
                    )

    def test_longer_series_amplifies_favorite(self) -> None:
        self.assertGreater(series_win_probability(0.7, 5), series_win_probability(0.7, 3))
        self.assertGreater(series_win_probability(0.7, 3), 0.7)

    def test_invalid_inputs_are_rejected(self) -> None:
        for p in (-0.01, 1.01, 10 ** 1000, math.nan, math.inf, -math.inf, "0.7", None, True):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    series_win_probability(p, 3)
        for best_of in (0, 2, 4, 7, 3.0, "3", None, True):
            with self.subTest(best_of=best_of):
                with self.assertRaises(ValueError):
                    series_win_probability(0.7, best_of)


if __name__ == "__main__":
    unittest.main()
