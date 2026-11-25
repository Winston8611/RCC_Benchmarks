"""Problem registry for the RCC and TP definitions used in the study.

The public module imports the standalone TP definitions from tp_benchmarks.
Historical machine-readable identifiers are retained so that saved result
metadata remain comparable. They are identifiers, not publication titles.

The registry intentionally contains only deterministic, bi-objective,
unconstrained problems that implement the common ``evaluate`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

try:
    from .tp_benchmarks import get_problem as get_tp_problem
    from .rcc_benchmarks import get_problem as get_rcc_problem
except ImportError:  # running as top-level modules, or via rcc_bench.py
    from tp_benchmarks import get_problem as get_tp_problem
    from rcc_benchmarks import get_problem as get_rcc_problem


# Kept stable for compatibility with the execution records.
REGISTRY_VERSION = "rcc-bench-type1-v1"


@dataclass(frozen=True)
class ProblemSpec:
    """Frozen metadata for one benchmark family."""

    name: str
    suite: str
    version: str
    default_n_var: int
    n_obj: int
    bounds_description: str
    constructor: Callable[[int], object]

    def build(self, n_var: int | None = None):
        n_var = self.default_n_var if n_var is None else int(n_var)
        if n_var < 2:
            raise ValueError("n_var must be at least 2")
        problem = self.constructor(n_var)
        _validate_problem(problem, self.name)
        return problem


def _tp(n_var: int):
    return get_tp_problem("TP1", n_var=n_var)  # replaced by family wrapper below


def _make_tp(name: str, n_var: int):
    return get_tp_problem(name, n_var=n_var)


def _make_rcc(name: str, n_var: int):
    return get_rcc_problem(name, n_var=n_var)


def _validate_problem(problem, name: str) -> None:
    required = ("n_var", "n_obj", "xl", "xu", "evaluate")
    missing = [attr for attr in required if not hasattr(problem, attr)]
    if missing:
        raise TypeError(f"{name} is missing required attributes: {missing}")
    if int(problem.n_obj) != 2:
        raise ValueError(f"{name} must be bi-objective")
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    if xl.shape != (int(problem.n_var),) or xu.shape != xl.shape:
        raise ValueError(f"{name} has invalid bounds shape")
    if not np.all(np.isfinite(xl)) or not np.all(np.isfinite(xu)):
        raise ValueError(f"{name} has non-finite bounds")
    if np.any(xu <= xl):
        raise ValueError(f"{name} has non-positive variable range")
    # A one-row smoke evaluation catches accidental dimension mismatches in a
    # registry entry before an expensive run starts.
    x = (xl + xu) / 2.0
    y = np.asarray(problem.evaluate(x), dtype=float)
    if y.shape != (1, 2) or not np.all(np.isfinite(y)):
        raise ValueError(f"{name} failed the objective-interface smoke test")


def _tp_constructor(name: str):
    return lambda n_var: _make_tp(name, n_var)


def _rcc_constructor(name: str):
    return lambda n_var: _make_rcc(name, n_var)


PROBLEM_SPECS: Dict[str, ProblemSpec] = {}
for _name in (f"TP{i}" for i in range(1, 10)):
    if _name in {"TP8", "TP9"}:
        _bounds = "TP8: [0,1]^2 x [-1,1]^(D-2); TP9: [0,1] x [-0.15,1] x [-1,1]^(D-2)"
    elif _name in {"TP6", "TP7"}:
        _bounds = "[0,1] x [-1,1]^(D-1)"
    else:
        _bounds = "[0,1]^D"
    PROBLEM_SPECS[_name] = ProblemSpec(
        name=_name,
        suite="TP",
        version=(
            "gaspar-cunha-2014-python-v1"
            if _name in {"TP1", "TP2", "TP3", "TP4", "TP5"}
            else "deb-gupta-2006-python-v1"
        ),
        default_n_var=10,
        n_obj=2,
        bounds_description=_bounds,
        constructor=_tp_constructor(_name),
    )

for _name in (f"RCC{i}" for i in range(1, 7)):
    _bounds = "RCC3: [0,1]^D" if _name == "RCC3" else "[0,1] x [-1,1]^(D-1)"
    PROBLEM_SPECS[_name] = ProblemSpec(
        name=_name,
        suite="RCC",
        version="rcc-original-v1",
        default_n_var=10,
        n_obj=2,
        bounds_description=_bounds,
        constructor=_rcc_constructor(_name),
    )

PROBLEM_SPECS["RCC7"] = ProblemSpec(
    name="RCC7",
    suite="RCC",
    version="rcc-bench-localized-v1",
    default_n_var=10,
    n_obj=2,
    bounds_description="[0,1] x [-1,1]^(D-1)",
    constructor=lambda n_var: get_rcc_problem("RCC7", n_var=n_var),
)

def get_problem(name: str, n_var: int | None = None):
    """Build a problem from the frozen registry."""
    key = str(name).upper()
    if key not in PROBLEM_SPECS:
        valid = ", ".join(PROBLEM_SPECS)
        raise ValueError(f"Unknown problem {name!r}; choose from {valid}")
    return PROBLEM_SPECS[key].build(n_var)


def get_spec(name: str) -> ProblemSpec:
    """Return immutable metadata without constructing the problem."""
    key = str(name).upper()
    if key not in PROBLEM_SPECS:
        raise KeyError(key)
    return PROBLEM_SPECS[key]


def names(suite: str | None = None) -> Tuple[str, ...]:
    """List registered names, optionally restricted to TP or RCC."""
    if suite is None:
        return tuple(PROBLEM_SPECS)
    suite = str(suite).upper()
    return tuple(k for k, spec in PROBLEM_SPECS.items() if spec.suite == suite)


__all__ = ["REGISTRY_VERSION", "ProblemSpec", "PROBLEM_SPECS", "get_problem", "get_spec", "names"]
