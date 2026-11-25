"""Mechanism-controlled RCC benchmark family.

The RCC problems share one construction kernel.  The first variable ``t`` is
the front coordinate and the remaining variables ``u`` describe a state in
which a narrow (fragile) basin competes with a broad (robust) basin.  The
family members declare their primary construction changes so that the
resulting candidate-state comparisons can be interpreted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import numpy as np


def _as_matrix(x: np.ndarray, n_var: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n_var:
        raise ValueError(f"expected (n_var,) or (N, {n_var}), got {arr.shape}")
    return arr


@dataclass
class BaseProblem:
    """Base class for all bi-objective RCC instances."""

    n_var: int = 10

    def __post_init__(self) -> None:
        if self.n_var < self.min_n_var:
            raise ValueError(
                f"{self.__class__.__name__} requires n_var >= {self.min_n_var}, "
                f"got {self.n_var}"
            )
        self.n_obj = 2
        self.xl, self.xu = self._bounds()
        self.name = self.__class__.__name__

    @property
    def min_n_var(self) -> int:
        return 2

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0] + [-1.0] * (self.n_var - 1)), np.ones(self.n_var)

    def _front_map(self, t: np.ndarray) -> np.ndarray:
        return t

    def _front_shape(self, f1: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(np.clip(f1, 0.0, 1.0))

    def _centres(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state_dim = self.n_var - 1
        broad = np.zeros((len(t), state_dim), dtype=float)
        fragile = np.full(
            (len(t), state_dim), np.sqrt(0.1 / state_dim), dtype=float
        )
        return broad, fragile

    def _matrix(self) -> np.ndarray:
        return np.eye(self.n_var - 1, dtype=float)

    def _gap(self, t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 0.08, dtype=float)

    def _basin_scales(self) -> tuple[float, float]:
        # Larger alpha means a narrower basin; the narrow scale is 0.005.
        return 1.0 / (0.005**2), 1.0

    def _state_distance(
        self, u: np.ndarray, centres: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        d = u - centres
        return np.einsum("ni,ij,nj->n", d, matrix, d) / np.trace(matrix)

    def _state_value(self, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        broad, fragile = self._centres(t)
        matrix = self._matrix()
        r_broad = self._state_distance(u, broad, matrix)
        r_fragile = self._state_distance(u, fragile, matrix)
        alpha_fragile, alpha_broad = self._basin_scales()
        return np.minimum(
            alpha_fragile * r_fragile,
            self._gap(t) + alpha_broad * r_broad,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = _as_matrix(x, self.n_var)
        if np.any(x < self.xl) or np.any(x > self.xu):
            raise ValueError("decision vectors are outside RCC bounds")
        t = x[:, 0]
        f1 = self._front_map(t)
        state = self._state_value(t, x[:, 1:])
        f2 = (1.0 + state) * self._front_shape(f1)
        return np.column_stack((f1, f2))


class RCC1(BaseProblem):
    """Baseline narrow-versus-broad basin conflict."""


class RCC2(RCC1):
    """RCC1 with only the unperturbed front geometry changed."""

    def _front_shape(self, f1: np.ndarray) -> np.ndarray:
        return 1.0 - np.asarray(f1) ** 2


class RCC3(RCC1):
    """RCC1 with a front-coordinate-dependent linkage manifold."""

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_var), np.ones(self.n_var)

    def _centres(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        broad = np.full((len(t), self.n_var - 1), 0.5, dtype=float) * t[:, None]
        fragile = broad + np.sqrt(0.1 / (self.n_var - 1))
        return broad, fragile


class RCC4(RCC1):
    """RCC1 with a larger conflict gap between the two basin minima."""

    def _gap(self, t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 0.20, dtype=float)


class RCC5(RCC1):
    """RCC1 with a non-uniform front-coordinate density map."""

    def _front_map(self, t: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-4.0 * t)


class RCC6(RCC1):
    """RCC1 with a positive-definite coupled state metric."""

    @property
    def min_n_var(self) -> int:
        return 3

    def _matrix(self) -> np.ndarray:
        d = self.n_var - 1
        matrix = np.eye(d, dtype=float)
        matrix += 0.20 * (np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1))
        # The eigenvalues remain positive for this tridiagonal construction.
        return matrix


class RCC7(RCC1):
    """Localized RCC conflict on one continuous unperturbed Pareto front.

    RCC7 is a member of RCC-Bench.  The gap between the fragile and broad
    states is activated by a smooth window over a proper front interval; the
    window is defined on all real ``t`` and is merely concentrated on
    ``[0.25, 0.45]``.
    """

    q_f: float = 0.10
    window_left: float = 0.25
    window_right: float = 0.45
    window_temperature: float = 0.015

    # Compatibility aliases retained for reference-generation scripts.  They
    # are construction parameters, not additional public problem names.
    baseline_slope: float = 5.0
    trap_amplitude: float = 1.0
    sigma: float = 0.005

    def _window(self, t: np.ndarray) -> np.ndarray:
        temp = self.window_temperature
        left = 1.0 / (1.0 + np.exp(-(t - self.window_left) / temp))
        right = 1.0 / (1.0 + np.exp((t - self.window_right) / temp))
        return left * right

    def _gap(self, t: np.ndarray) -> np.ndarray:
        broad, fragile = self._offsets(t)
        return np.abs(broad - fragile)

    def _offsets(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        window = self._window(t)
        broad_active = np.maximum(2.0 * window - 1.0, 0.0)
        fragile_inactive = np.maximum(1.0 - 2.0 * window, 0.0)
        broad = 0.12 * broad_active**2
        fragile = 0.20 * fragile_inactive**2
        return broad, fragile

    def _state_value(self, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        broad, fragile = self._centres(t)
        matrix = self._matrix()
        r_broad = self._state_distance(u, broad, matrix)
        r_fragile = self._state_distance(u, fragile, matrix)
        alpha_fragile, alpha_broad = self._basin_scales()
        broad_offset, fragile_offset = self._offsets(t)
        broad_cost = broad_offset + alpha_broad * r_broad
        fragile_cost = fragile_offset + alpha_fragile * r_fragile
        return np.minimum(broad_cost, fragile_cost)

    def _centres(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        broad, fragile = super()._centres(t)
        fragile[:] = np.sqrt(self.q_f / (self.n_var - 1))
        return broad, fragile

    def window_value(self, t: np.ndarray | float) -> np.ndarray:
        return self._window(np.asarray(t, dtype=float))

    def nominal_state_values(self, t: np.ndarray | float) -> dict[str, np.ndarray]:
        t = np.asarray(t, dtype=float)
        broad, fragile = self._offsets(t)
        return {
            "broad_q0": broad,
            "fragile_qf": fragile,
            "trap_qf": fragile,
            "window": self._window(t),
            "gap": self._gap(t),
        }


PROBLEM_MAP: Dict[str, Type[BaseProblem]] = {
    "RCC1": RCC1,
    "RCC2": RCC2,
    "RCC3": RCC3,
    "RCC4": RCC4,
    "RCC5": RCC5,
    "RCC6": RCC6,
    "RCC7": RCC7,
}


def get_problem(name: str, n_var: int = 10) -> BaseProblem:
    key = str(name).upper()
    if key not in PROBLEM_MAP:
        raise ValueError(f"Unknown problem '{name}'. Available: {', '.join(PROBLEM_MAP)}")
    return PROBLEM_MAP[key](n_var=n_var)


__all__ = ["BaseProblem", *[f"RCC{i}" for i in range(1, 8)], "PROBLEM_MAP", "get_problem"]
