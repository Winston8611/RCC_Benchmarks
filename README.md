# RCC-Bench

**RCC-Bench: A Benchmark Suite for Robustness–Convergence Conflicts in Evolutionary Multi-objective Optimization**

Wenxiang Jiang<sup>1</sup>, Fangrui Bi<sup>2</sup>, and Lihong Xu<sup>1,*</sup>

<sup>1</sup> School of Electronic and Information Engineering, Tongji University, Shanghai 201804, China  
<sup>2</sup> Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University, Shanghai 201210, China  

\* Corresponding author: Lihong Xu (`xulihong@tongji.edu.cn`)

This repository has been accepted for publication in *Complex & Intelligent Systems*. Full bibliographic details will be updated after the article is indexed in Web of Science.

This repository distributes the benchmark definitions only: RCC1–RCC7 and the TP1–TP9 problems used with them. Comparison-algorithm implementations are outside this release. The numerical result archive is deposited separately on Figshare: <https://doi.org/10.6084/m9.figshare.33320715>.

## Overview

RCC-Bench is a diagnostic benchmark for Type-I robust multi-objective evolutionary optimization under decision-variable perturbations. In the Type-I model of Deb and Gupta (2006), each objective is replaced by its mean in a neighborhood of the decision. A decision that is attractive on the unperturbed landscape can then become inferior once that neighborhood is taken into account. The suite is built to make this robustness–convergence conflict explicit, and to connect a search method’s basin decisions with the quality of the set it returns.

The study pairs seven new bi-objective problems, RCC1–RCC7, with nine existing landscapes, TP1–TP9. TP1–TP5 follow Gaspar-Cunha, Ferreira, and Recio (2014). TP6–TP9 are Problems 1–4 of Deb and Gupta (2006), in that order. The reported experiments cover nine evolutionary methods, spanning robust methods, nominal multi-objective evolutionary algorithms, and LR-MOEA, together with the engineering model RE21 and a small-budget surrogate comparison.

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

## What the controlled comparisons show

The main controlled comparison starts from shared decisions in the fragile region, or from paired samples of both basin centers. With the same initial decisions available to every method, the suite separates basin retention from the HV and IGD+ of the final effective set.

On the displaced starts of RCC1–RCC6, the nominal algorithms NSGA-II and SMS-EMOA retain a high fraction of fragile-basin solutions (median retention 73%–99.5%). NSGA-II-DT1, which evaluates the Type-I mean directly, leaves that basin (0%–3.5%). All 42 pairwise comparisons of this retention outcome are significant after Holm adjustment (`p = 0.00433`).

The same controlled setting also separates methods by geometry. On RCC2, NSGA-II-DT1 leads NSGA-II and SMS-EMOA on both HV and IGD+ (`p = 0.00433`). RCC7 places a local improvement inside the conflict interval against coverage of the rest of the front. Across the 21 RCC problem–initialization combinations, LR-MOEA is weaker overall than NSGA-II, MOEA/D, and SMS-EMOA: 115 of the 126 indicator comparisons are significant.

Rankings also move with the experiment settings. As the perturbation ratio `ρ` increases from 0 to 0.10, the HV leader on the RCC problems is most often RMOEA-SuR for `ρ ≤ 0.025` and most often LR-MOEA for `ρ ≥ 0.05`. At `D ∈ {10, 15, 20}`, a fixed evaluation budget lowers rank consistency; raising the budget to `2000D` improves most of the tested methods.

Read together, these results describe a diagnostic use. Given a start at which both basins compete, RCC-Bench exposes how a search method trades basin retention, perturbation-sampling cost, front coverage, and evaluation budget. Random-start rankings, and the RE21 engineering comparison, address a different question: they do not by themselves produce a stable gap between robust methods, nominal methods, and the engineering case.

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

## Citation

Please cite the paper if you use these definitions. The entry below will be replaced with the complete Web of Science record after indexing.

```bibtex
@article{jiang2026rccbench,
  author  = {Jiang, Wenxiang and Bi, Fangrui and Xu, Lihong},
  title   = {{RCC-Bench}: A Benchmark Suite for Robustness--Convergence Conflicts in Evolutionary Multi-objective Optimization},
  journal = {Complex \& Intelligent Systems},
  year    = {2026},
  note    = {Accepted for publication; full citation details will be updated upon Web of Science indexing}
}
```

The TP landscapes are included so that the RCC problems can be used beside the collection studied in the paper. Their original sources are:

```bibtex
@article{deb2006introducing,
  author  = {Deb, Kalyanmoy and Gupta, Himanshu},
  title   = {Introducing Robustness in Multiobjective Optimization},
  journal = {Evolutionary Computation},
  volume  = {14},
  number  = {4},
  pages   = {463--494},
  year    = {2006},
  doi     = {10.1162/evco.2006.14.4.463}
}

@article{gasparcunha2014evolutionary,
  author  = {Gaspar-Cunha, Ant{\'o}nio and Ferreira, Jose and Recio, Gustavo},
  title   = {Evolutionary Robustness Analysis for Multi-objective Optimization: Benchmark Problems},
  journal = {Structural and Multidisciplinary Optimization},
  volume  = {49},
  number  = {5},
  pages   = {771--793},
  year    = {2014},
  doi     = {10.1007/s00158-013-1010-x}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62373286).
