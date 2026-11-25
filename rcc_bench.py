"""Import alias for the RCC-Bench repository root.

``RCC-Bench`` is not a legal Python identifier. This module provides the
stable import ``from rcc_bench import get_problem`` when the repository
root is on ``sys.path``.
"""

from problem_registry import (
    PROBLEM_SPECS,
    REGISTRY_VERSION,
    ProblemSpec,
    get_problem,
    get_spec,
    names,
)
from rcc_benchmarks import (
    RCC1,
    RCC2,
    RCC3,
    RCC4,
    RCC5,
    RCC6,
    RCC7,
)
from tp_benchmarks import (
    TP1,
    TP2,
    TP3,
    TP4,
    TP5,
    TP6,
    TP7,
    TP8,
    TP9,
)

__all__ = [
    "REGISTRY_VERSION",
    "ProblemSpec",
    "PROBLEM_SPECS",
    "get_problem",
    "get_spec",
    "names",
    "RCC1",
    "RCC2",
    "RCC3",
    "RCC4",
    "RCC5",
    "RCC6",
    "RCC7",
    "TP1",
    "TP2",
    "TP3",
    "TP4",
    "TP5",
    "TP6",
    "TP7",
    "TP8",
    "TP9",
]
