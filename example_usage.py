#!/usr/bin/env python3
"""Evaluate one interior decision on every registered RCC and TP problem."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem_registry import get_problem, names  # noqa: E402
from rcc_bench import get_problem as get_problem_alias  # noqa: E402


def main() -> None:
    registered = names()
    expected = tuple(f"TP{i}" for i in range(1, 10)) + tuple(
        f"RCC{i}" for i in range(1, 8)
    )
    if registered != expected:
        raise SystemExit(f"unexpected registry order: {registered}")

    print(f"{'problem':<8} {'D':>3} {'f1':>14} {'f2':>14}")
    for name in registered:
        problem = get_problem(name, n_var=5)
        alias = get_problem_alias(name, n_var=5)
        x = (np.asarray(problem.xl, dtype=float) + np.asarray(problem.xu, dtype=float)) / 2.0
        values = np.asarray(problem.evaluate(x), dtype=float)
        alias_values = np.asarray(alias.evaluate(x), dtype=float)
        if values.shape != (1, 2) or not np.all(np.isfinite(values)):
            raise SystemExit(f"{name} returned {values!r}")
        if not np.allclose(values, alias_values):
            raise SystemExit(f"{name} disagrees between import paths")
        print(f"{name:<8} {problem.n_var:3d} {values[0, 0]:14.6f} {values[0, 1]:14.6f}")

    batch = get_problem("RCC7", n_var=5)
    samples = np.linspace(batch.xl, batch.xu, 4)
    batch_values = np.asarray(batch.evaluate(samples), dtype=float)
    if batch_values.shape != (4, 2) or not np.all(np.isfinite(batch_values)):
        raise SystemExit(f"RCC7 batch evaluation returned {batch_values!r}")
    print(f"registered {len(registered)} problems; RCC7 batch shape {batch_values.shape}")


if __name__ == "__main__":
    main()
