"""TP1--TP9 robustness benchmark definitions used in the study.

TP1--TP5 follow Gaspar-Cunha et al. (2014), while TP6--TP9 are
the four Deb--Gupta (2006) Type-I cases re-indexed by that later work.  The
variable bounds follow the public MATLAB implementation distributed with the
suite:

* TP1--TP5: ``[0, 1]^D``;
* TP6--TP7: ``[0, 1] x [-1, 1]^(D-1)``;
* TP8: ``[0, 1]^2 x [-1, 1]^(D-2)``;
* TP9: ``[0, 1] x [-0.15, 1] x [-1, 1]^(D-2)``.

All problems expose ``n_var``, ``n_obj``, ``xl``, ``xu`` and
``evaluate(X) -> (N, n_obj)``.  The explicit bounds are part of the benchmark
definition and must not be inferred from an algorithm output.
"""
import numpy as np


class BaseProblem:
    def __init__(self, n_var=10):
        self.name = self.__class__.__name__
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)

    def _set_bounds(self, lower, upper):
        """Set and validate the decision-space bounds for a TP instance."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != (self.n_var,) or upper.shape != (self.n_var,):
            raise ValueError("TP bounds must have shape (n_var,)")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("TP bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("TP upper bounds must exceed lower bounds")
        self.xl, self.xu = lower, upper

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._evaluate(X)

    def _evaluate(self, X):
        raise NotImplementedError


class TP1(BaseProblem):
    def _evaluate(self, X):
        V = self.n_var
        f0 = X[:, 0]
        h = ((X[:, 0] - 0.6) ** 3 - 0.4 ** 3) / (-0.6 ** 3 - 0.4 ** 3)
        g = (X[:, 1:].sum(axis=1) / (V - 1)) ** 2
        s = 1.0 / (0.2 + X[:, 0])
        return np.column_stack([f0, h + g * s])


class TP2(BaseProblem):
    def _evaluate(self, X):
        V = self.n_var
        f0 = np.cos(np.pi * X[:, 0] / 2)
        g = 1 + 10 * (X[:, 1:].sum(axis=1) / (V - 1))
        return np.column_stack([f0, g * np.sin(np.pi * X[:, 0] / 2)])


class TP3(BaseProblem):
    def _evaluate(self, X):
        V = self.n_var
        f0 = 1 - X[:, 0] ** 2
        g = 1 + 10 * (X[:, 1:].sum(axis=1) / (V - 1))
        return np.column_stack([f0, g * np.sin(np.pi * X[:, 0] / 2)])


class TP4(BaseProblem):
    def _evaluate(self, X):
        V = self.n_var
        f0 = (np.exp(X[:, 0]) - 1) / (np.exp(1) - 1)
        g = 1 + 10 * (X[:, 1:].sum(axis=1) / (V - 1))
        f1 = g * ((np.sin(4 * np.pi * X[:, 0]) - 15 * X[:, 0]) / 15 + 1)
        return np.column_stack([f0, f1])


class TP5(BaseProblem):
    def _evaluate(self, X):
        V = self.n_var
        f0 = X[:, 0]
        g = 1 + 10 * (X[:, 1:].sum(axis=1) / (V - 1))
        f1 = g * ((np.sin(4 * np.pi * X[:, 0]) - 15 * X[:, 0]) / 15 + 1)
        return np.column_stack([f0, f1])


class TP6(BaseProblem):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self._set_bounds(
            np.array([0.0] + [-1.0] * (n_var - 1)),
            np.ones(n_var),
        )

    def _evaluate(self, X):
        f0 = X[:, 0]
        h = 1 - X[:, 0] ** 2
        g = (10 + X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:])).sum(axis=1)
        s = 1.0 / (0.2 + X[:, 0]) + X[:, 0] ** 2
        return np.column_stack([f0, h + g * s])


class TP7(BaseProblem):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self._set_bounds(
            np.array([0.0] + [-1.0] * (n_var - 1)),
            np.ones(n_var),
        )

    def _evaluate(self, X):
        f0 = X[:, 0]
        h = 1 - X[:, 0] ** 2
        g = (10 + X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:])).sum(axis=1)
        s = 1.0 / (0.2 + X[:, 0]) + 10 * X[:, 0] ** 2
        return np.column_stack([f0, h + g * s])


class TP8(BaseProblem):
    def __init__(self, n_var=10):
        if n_var < 3:
            raise ValueError("TP8 requires n_var >= 3")
        super().__init__(n_var)
        self._set_bounds(
            np.array([0.0, 0.0] + [-1.0] * (n_var - 2)),
            np.ones(n_var),
        )

    def _evaluate(self, X):
        f0 = X[:, 0]
        h = 2 - 0.8 * np.exp(-((X[:, 1] - 0.35) / 0.25) ** 2) - np.exp(-((X[:, 1] - 0.85) / 0.03) ** 2)
        g = (50 * X[:, 2:] ** 2).sum(axis=1)
        s = 1 - np.sqrt(X[:, 0])
        return np.column_stack([f0, h * (g + s)])


class TP9(BaseProblem):
    def __init__(self, n_var=10):
        if n_var < 3:
            raise ValueError("TP9 requires n_var >= 3")
        super().__init__(n_var)
        self._set_bounds(
            np.array([0.0, -0.15] + [-1.0] * (n_var - 2)),
            np.ones(n_var),
        )

    def _evaluate(self, X):
        f0 = X[:, 0]
        h = 2 - X[:, 0] - 0.8 * np.exp(-((X[:, 0] + X[:, 1] - 0.35) / 0.25) ** 2) - np.exp(-((X[:, 0] + X[:, 1] - 0.85) / 0.03) ** 2)
        g = (50 * X[:, 2:] ** 2).sum(axis=1)
        s = 1 - np.sqrt(X[:, 0])
        return np.column_stack([f0, h * (g + s)])


# Keep the benchmark registry limited to implemented deterministic problems.
# TP1--TP5 follow Gaspar-Cunha et al. (2014); TP6--TP9 are the four
# Deb--Gupta (2006) Type-I cases as re-indexed by that later work.
# Application instances must be added only after their model and reference
# outputs are part of the reproducibility manifest.
PROBLEMS = {
    'TP1': TP1, 'TP2': TP2, 'TP3': TP3, 'TP4': TP4, 'TP5': TP5,
    'TP6': TP6, 'TP7': TP7, 'TP8': TP8, 'TP9': TP9,
}

ALL_PROBLEMS = tuple(PROBLEMS)


def get_problem(name, n_var=10):
    """Construct one canonical benchmark problem by its case-insensitive name."""
    key = str(name).upper()
    if key not in PROBLEMS:
        raise ValueError(f"Unknown problem {name!r}; choose from {ALL_PROBLEMS}")
    return PROBLEMS[key](n_var=n_var)
