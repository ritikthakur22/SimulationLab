"""Unit tests for SimulationLab mathematical models and algorithms.

Tests cover:
- Lab 1: Chemical reaction kinetics (Euler method numerical simulation)
- Lab 2: Linear Congruential Generator (LCG) and Chi-Square goodness-of-fit test
- Lab 3: Monte Carlo Pi estimation and geometric probability
- Lab 4: Markov Decision Process (MDP) and Q-Learning Bellman updates
"""

from __future__ import annotations

import math
import random
import unittest
import numpy as np


class ChemicalKineticsTests(unittest.TestCase):
    def test_euler_step_reaction1(self) -> None:
        """Verify chemical equilibrium kinetics Euler update step."""
        k1 = 0.03
        k2 = 0.01
        dt = 0.1
        steps = 30

        c1 = [50.0]
        c2 = [25.0]
        c3 = [0.0]

        for i in range(steps):
            c1_next = c1[i] + (k2 * c3[i] - k1 * c1[i] * c2[i]) * dt
            c2_next = c2[i] + (k2 * c3[i] - k1 * c1[i] * c2[i]) * dt
            c3_next = c3[i] + (2 * k1 * c1[i] * c2[i] - 2 * k2 * c3[i]) * dt

            c1.append(c1_next)
            c2.append(c2_next)
            c3.append(c3_next)

        self.assertEqual(len(c1), 31)
        self.assertGreater(c3[-1], c3[0])
        self.assertLess(c1[-1], c1[0])
        self.assertLess(c2[-1], c2[0])
        self.assertTrue(all(val >= 0 for val in c1))
        self.assertTrue(all(val >= 0 for val in c2))
        self.assertTrue(all(val >= 0 for val in c3))


class PseudoRandomLCGTests(unittest.TestCase):
    def test_lcg_generation_and_chi_square(self) -> None:
        """Verify LCG generates complete period and achieves perfect uniform distribution."""
        iterator = 256
        a = 29
        b = 37
        p = 256
        n = 8
        ei = iterator / n

        r = np.ones(shape=(iterator), dtype=int)
        c = np.zeros(shape=(n), dtype=int)
        r[0] = 17

        for i in range(iterator - 1):
            r[i + 1] = (r[i] * a + b) % p

        # Check full period
        self.assertEqual(len(set(r)), 256)

        # Binning into 8 bins of size 32
        for val in r:
            bin_idx = min(val // 32, n - 1)
            c[bin_idx] += 1

        # In this specific configuration, each bin receives exactly 32 values
        self.assertTrue(np.all(c == 32))

        # Chi-Square statistic
        chi_square_stat = sum(((count - ei) ** 2) / ei for count in c)
        self.assertEqual(chi_square_stat, 0.0)


class MonteCarloPiTests(unittest.TestCase):
    def test_geometry_point_in_circle(self) -> None:
        """Verify geometric boundary test for inscribed circle."""
        length = 1.0
        radius_sq = (length / 2) ** 2

        # Origin is inside
        self.assertTrue((0.0) ** 2 + (0.0) ** 2 <= radius_sq)
        # Boundary points
        self.assertTrue((0.5) ** 2 + (0.0) ** 2 <= radius_sq)
        self.assertTrue((0.0) ** 2 + (-0.5) ** 2 <= radius_sq)
        # Corner of square is outside
        self.assertFalse((0.5) ** 2 + (0.5) ** 2 <= radius_sq)
        self.assertFalse((-0.5) ** 2 + (-0.5) ** 2 <= radius_sq)

    def test_pi_estimation_convergence(self) -> None:
        """Verify Monte Carlo estimation converges towards mathematical Pi."""
        random.seed(42)
        total_drops = 15000
        length_of_field = 1.0
        radius_sq = (length_of_field / 2) ** 2

        inside = 0
        for _ in range(total_drops):
            x = (0.5 - random.random()) * length_of_field
            y = (0.5 - random.random()) * length_of_field
            if x * x + y * y <= radius_sq:
                inside += 1

        pi_estimate = 4.0 * inside / total_drops
        self.assertAlmostEqual(pi_estimate, math.pi, delta=0.08)


class QLearningGridWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows, self.cols = 4, 4
        self.num_states = self.rows * self.cols
        self.goal = 15
        self.obstacles = [5, 10]

    def _next_state(self, state: int, action: int) -> int:
        r = state // self.cols
        c = state % self.cols

        if action == 0 and r > 0:  # UP
            r -= 1
        elif action == 1 and r < self.rows - 1:  # DOWN
            r += 1
        elif action == 2 and c > 0:  # LEFT
            c -= 1
        elif action == 3 and c < self.cols - 1:  # RIGHT
            c += 1

        return r * self.cols + c

    def _reward(self, state: int) -> int:
        if state == self.goal:
            return 10
        elif state in self.obstacles:
            return -10
        else:
            return -1

    def test_grid_boundaries(self) -> None:
        """Verify state transitions remain within 4x4 grid."""
        # Top-left corner (0)
        self.assertEqual(self._next_state(0, 0), 0)  # UP -> stays 0
        self.assertEqual(self._next_state(0, 2), 0)  # LEFT -> stays 0
        self.assertEqual(self._next_state(0, 1), 4)  # DOWN -> moves to 4
        self.assertEqual(self._next_state(0, 3), 1)  # RIGHT -> moves to 1

        # Bottom-right corner (15)
        self.assertEqual(self._next_state(15, 1), 15)  # DOWN -> stays 15
        self.assertEqual(self._next_state(15, 3), 15)  # RIGHT -> stays 15

    def test_reward_mapping(self) -> None:
        """Verify rewards for goal, obstacles, and intermediate transitions."""
        self.assertEqual(self._reward(15), 10)
        self.assertEqual(self._reward(5), -10)
        self.assertEqual(self._reward(10), -10)
        self.assertEqual(self._reward(0), -1)
        self.assertEqual(self._reward(7), -1)

    def test_bellman_update_calculation(self) -> None:
        """Verify exact Q-value Bellman update matching Lab 4 manual episode trace."""
        alpha = 0.5
        gamma = 0.9

        # S0 -> S1 (Normal step)
        q_s0_r = 0.0 + alpha * (-1 + gamma * 0.0 - 0.0)
        self.assertEqual(q_s0_r, -0.5)

        # S1 -> S5 (Obstacle step)
        q_s1_d = 0.0 + alpha * (-10 + gamma * 0.0 - 0.0)
        self.assertEqual(q_s1_d, -5.0)

        # S14 -> S15 (Goal step)
        q_s14_r = 0.0 + alpha * (10 + gamma * 0.0 - 0.0)
        self.assertEqual(q_s14_r, 5.0)

        # Backpropagation: S13 -> S14 where Q(S14, R) is 5.0
        prev_q = -0.5
        q_s13_r = prev_q + alpha * (-1 + gamma * 5.0 - prev_q)
        self.assertEqual(q_s13_r, 1.5)


if __name__ == "__main__":
    unittest.main()
