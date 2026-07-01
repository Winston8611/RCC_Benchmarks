"""
RCC benchmark suite for Robust Multi-objective Optimization (RMOO).

The Robustness-Convergence Conflict (RCC) suite is designed to test whether
an optimizer can move away from fragile convergence-optimal regions and locate
solutions that are more robust under decision-variable perturbations.

All problems are bi-objective minimization problems. The first decision
variable controls the Pareto-front shape, while the remaining variables define
the robustness-related landscape.

Reference design idea:
    Sharp Peaks vs. Flat Plateaus

Dependencies:
    numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import numpy as np


@dataclass
class BaseProblem:
    """Base class for RCC test problems."""

    n_var: int = 10

    def __post_init__(self) -> None:
        if self.n_var < self.min_n_var:
            raise ValueError(
                f"{self.__class__.__name__} requires n_var >= {self.min_n_var}, "
                f"got {self.n_var}."
            )
        self.n_obj = 2
        self.xl, self.xu = self._bounds()

    @property
    def min_n_var(self) -> int:
        return 2

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([0.0] + [-1.0] * (self.n_var - 1))
        upper = np.ones(self.n_var)
        return lower, upper

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate objective values.

        Parameters
        ----------
        x : np.ndarray
            Decision vector with shape ``(n_var,)`` or decision matrix with
            shape ``(n_samples, n_var)``.

        Returns
        -------
        np.ndarray
            Objective matrix with shape ``(n_samples, 2)``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.n_var:
            raise ValueError(
                f"Expected input shape ({self.n_var},) or (n_samples, {self.n_var}), "
                f"got {x.shape}."
            )
        return self._evaluate(x)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RCC1(BaseProblem):
    """
    RCC1: Baseline robustness-convergence conflict.

    - Front shape: convex.
    - Fragile global optimum: a very narrow Gaussian trap at g_r = 0.1.
    - Robust local region: a wider, flatter region near g_r = 0.
    """

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g_r = np.sum(x[:, 1:] ** 2, axis=1)

        exp_term = np.exp(-((g_r - 0.1) ** 2) / (2.0 * (0.0005**2)))
        linear_term = 5.0 * g_r
        g = 1.0 - exp_term + linear_term

        f2 = (1.0 + g) * (1.0 - np.sqrt(f1))
        return np.column_stack((f1, f2))


class RCC2(BaseProblem):
    """
    RCC2: Concave-front variant with swapped fragile/robust positions.

    - Front shape: concave.
    - Fragile global optimum: a narrow boundary well at g_r = 0.
    - Robust local region: a wider quadratic basin near g_r = 0.1.
    """

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g_r = np.sum(x[:, 1:] ** 2, axis=1)

        robust_basin = 0.8 + 2.0 * (g_r - 0.1) ** 2
        fragile_well = 0.8 * np.exp(-(g_r**2) / (2.0 * (0.0005**2)))
        g = robust_basin - fragile_well

        f2 = (1.0 + g) * (1.0 - f1**2)
        return np.column_stack((f1, f2))


class RCC3(BaseProblem):
    """
    RCC3: Shifted-valley variant with variable linkage.

    The fragile and robust regions move with x_1, so the optimizer must track
    a position-dependent robust region instead of converging to a fixed
    location.
    """

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_var), np.ones(self.n_var)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        shifted_vars = x[:, 1:] - 0.5 * x[:, 0:1]
        g_r = np.sum(shifted_vars**2, axis=1)

        exp_term = np.exp(-((g_r - 0.1) ** 2) / (2.0 * (0.0005**2)))
        linear_term = 5.0 * g_r
        g = 1.0 - exp_term + linear_term

        f2 = (1.0 + g) * (1.0 - np.sqrt(f1))
        return np.column_stack((f1, f2))


class RCC4(BaseProblem):
    """
    RCC4: Twin-basin landscape with large gradient contrast.

    - State variable: mean of x_2, ..., x_n.
    - Fragile basin: deep and steep, centered at z = -0.5.
    - Robust basin: shallower and flatter, centered at z = 0.5.
    """

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        z = np.mean(x[:, 1:], axis=1)

        fragile_basin = 500.0 * (z + 0.5) ** 2
        robust_basin = 0.1 + 10.0 * (z - 0.5) ** 2
        g = np.minimum(fragile_basin, robust_basin)
        g = np.maximum(g, 0.0)

        f2 = (1.0 + g) * (1.0 - np.sqrt(f1))
        return np.column_stack((f1, f2))


class RCC5(BaseProblem):
    """
    RCC5: Twin-basin landscape with non-uniform f1 mapping.

    This problem adds an objective-space density bias through
    f1 = 1 - exp(-4*x1) and strengthens the attraction of the fragile basin.
    """

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        z = np.mean(x[:, 1:], axis=1)

        f1 = 1.0 - np.exp(-4.0 * x1)
        fragile_basin = 5000.0 * (z + 0.5) ** 2
        robust_basin = 0.2 + 50.0 * (z - 0.5) ** 2
        g = np.minimum(fragile_basin, robust_basin)
        g = np.maximum(g, 0.0)

        f2 = (1.0 + g) * (1.0 - np.sqrt(f1))
        return np.column_stack((f1, f2))


class RCC6(BaseProblem):
    """
    RCC6: Coupled Gaussian trap.

    This problem follows the RCC1 trap-vs-plateau structure, but the robustness
    variable is based on adjacent-variable differences. Small perturbations can
    destroy the required chain-like relationship among variables.
    """

    @property
    def min_n_var(self) -> int:
        return 3

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g_r = np.sum((x[:, 1:-1] - x[:, 2:]) ** 2, axis=1)

        exp_term = np.exp(-((g_r - 0.1) ** 2) / (2.0 * (0.0005**2)))
        linear_term = 5.0 * g_r
        g = 1.0 - exp_term + linear_term

        f2 = (1.0 + g) * (1.0 - np.sqrt(f1))
        return np.column_stack((f1, f2))


PROBLEM_MAP: Dict[str, Type[BaseProblem]] = {
    "RCC1": RCC1,
    "RCC2": RCC2,
    "RCC3": RCC3,
    "RCC4": RCC4,
    "RCC5": RCC5,
    "RCC6": RCC6,
}


def get_problem(name: str, n_var: int = 10) -> BaseProblem:
    """
    Create an RCC benchmark problem by name.

    Examples
    --------
    >>> problem = get_problem("RCC1", n_var=10)
    >>> x = np.random.uniform(problem.xl, problem.xu, size=(100, problem.n_var))
    >>> y = problem.evaluate(x)
    """
    key = name.upper()
    if key not in PROBLEM_MAP:
        valid = ", ".join(PROBLEM_MAP)
        raise ValueError(f"Unknown problem '{name}'. Available problems: {valid}.")
    return PROBLEM_MAP[key](n_var=n_var)


__all__ = [
    "BaseProblem",
    "RCC1",
    "RCC2",
    "RCC3",
    "RCC4",
    "RCC5",
    "RCC6",
    "PROBLEM_MAP",
    "get_problem",
]


if __name__ == "__main__":
    problem = get_problem("RCC1", n_var=10)
    decisions = np.random.uniform(problem.xl, problem.xu, size=(5, problem.n_var))
    objectives = problem.evaluate(decisions)
    print("Decision shape:", decisions.shape)
    print("Objective shape:", objectives.shape)
    print(objectives)
