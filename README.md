# RCC-Bench

**RCC-Bench: A Benchmark Suite for Robustness–Convergence Conflicts in Evolutionary Multi-objective Optimization**

This repository distributes the benchmark definitions only: RCC1–RCC7 and the TP1–TP9 problems used with them. The numerical result archive is deposited separately on Figshare: <https://doi.org/10.6084/m9.figshare.33320715>.

## Overview

RCC-Bench is a diagnostic benchmark for Type-I robust multi-objective evolutionary optimization under decision-variable perturbations. In the Type-I model of Deb and Gupta (2006), each objective is replaced by its mean in a neighborhood of the decision. A decision that is attractive on the unperturbed landscape can then become inferior once that neighborhood is taken into account. The suite is built to make this robustness–convergence conflict explicit, and to connect a search method’s basin decisions with the quality of the set it returns.

The study pairs seven new bi-objective problems, RCC1–RCC7, with nine existing landscapes, TP1–TP9. TP1–TP5 follow Gaspar-Cunha, Ferreira, and Recio (2014). TP6–TP9 are Problems 1–4 of Deb and Gupta (2006), in that order.

## Benchmark design

RCC1–RCC7 share one two-basin construction. At each position along the front, a narrow fragile basin is preferable under the nominal objectives and more sensitive to perturbation, while a broad basin can attain the better Type-I mean. The members change one primary construction factor at a time, so a change in measured behavior can be traced to that factor. A finite pre-search audit checks the declared fragile–broad comparison before any algorithm is run.

| Problem | Primary construction change |
| --- | --- |
| RCC1 | Baseline narrow-versus-broad basins |
| RCC2 | Front shape |
| RCC3 | Moving basin centers, with domain `[0, 1]^D` |
| RCC4 | Larger broad-basin offset |
| RCC5 | Non-uniform front-coordinate map |
| RCC6 | Coupled positive-definite distance metric (`D ≥ 3`) |
| RCC7 | Localized conflict on one interval of an otherwise continuous front |

Every problem is unconstrained and bi-objective, and every objective is minimized. A problem instance exposes `n_var`, `n_obj`, `xl`, `xu`, and `evaluate(X)`. For an input of shape `(N, D)`, or a single decision of shape `(D,)`, `evaluate` returns an objective array of shape `(N, 2)`.

Decision bounds are part of the definition:

| Family | Bounds |
| --- | --- |
| RCC1, RCC2, RCC4–RCC7 | `[0, 1] × [-1, 1]^(D-1)` |
| RCC3 | `[0, 1]^D` |
| TP1–TP5 | `[0, 1]^D` |
| TP6–TP7 | `[0, 1] × [-1, 1]^(D-1)` |
| TP8 (`D ≥ 3`) | `[0, 1]^2 × [-1, 1]^(D-2)` |
| TP9 (`D ≥ 3`) | `[0, 1] × [-0.15, 1] × [-1, 1]^(D-2)` |

The accompanying evaluation protocol, described in the paper, is held fixed across methods. It builds reference sets from the problem definitions, estimates each returned decision by an independent perturbation sample, keeps the non-dominated effective set, and scores that set with fixed-reference hypervolume (HV) and modified inverted generational distance (IGD+). Basin retention is recorded separately from final-set quality and front coverage.

## Repository contents

| File | Role |
| --- | --- |
| `rcc_benchmarks.py` | RCC1–RCC7 definitions |
| `tp_benchmarks.py` | TP1–TP9 definitions |
| `problem_registry.py` | Versioned registry and `get_problem` |
| `rcc_bench.py` | Import alias `from rcc_bench import get_problem` |
| `__init__.py` | Package exports |
| `example_usage.py` | Smoke evaluation of all 16 problems |
| `requirements.txt` | Runtime dependency (`numpy`) |

## Installation

Python 3.9 or later is sufficient. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example_usage.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Usage

Add this directory to `sys.path`, then construct a problem by name. Both import forms below refer to the same registry.

```python
import numpy as np
from problem_registry import get_problem, names
from rcc_bench import get_problem as get_rcc_bench_problem

print(names())  # TP1–TP9, then RCC1–RCC7

problem = get_problem("RCC1", n_var=5)
x = (problem.xl + problem.xu) / 2.0
f = problem.evaluate(x)          # shape (1, 2)

X = np.linspace(problem.xl, problem.xu, 8)
F = problem.evaluate(X)          # shape (8, 2)

same = get_rcc_bench_problem("TP6", n_var=10)
```

`get_problem(name, n_var=None)` accepts any registered name, ignoring case. The default dimension is 10. RCC6, TP8, and TP9 require at least three variables; every other problem requires at least two. Decisions outside `xl` and `xu` are rejected by the RCC problems.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62373286).
