# RCC Benchmark Suite

This repository provides the **Robustness-Convergence Conflict (RCC)** benchmark suite for Robust Multi-objective Optimization (RMOO).

The RCC suite is designed to evaluate whether an optimization algorithm can explicitly handle the conflict between:

- **Convergence-optimal but fragile solutions**: solutions that are attractive under noise-free evaluation but degrade severely under decision-variable perturbations.
- **Convergence-inferior but robust solutions**: solutions that are slightly worse in nominal performance but remain stable under perturbations.

The core design idea is:

> **Sharp Peaks vs. Flat Plateaus**

In each RCC problem, the global optimum in the deterministic landscape is intentionally placed in a narrow and fragile region, while a more robust local optimum is placed in a wider and flatter region.

## Files

- `rcc_benchmarks.py`: standalone implementation of RCC1-RCC6.

## Requirements

```bash
pip install numpy
```

## Quick Start

```python
import numpy as np
from rcc_benchmarks import get_problem

# Create a benchmark problem
problem = get_problem("RCC1", n_var=10)

# Generate random decision vectors within the search bounds
x = np.random.uniform(problem.xl, problem.xu, size=(100, problem.n_var))

# Evaluate objective values
f = problem.evaluate(x)

print(f.shape)  # (100, 2)
```

You can also instantiate a problem class directly:

```python
from rcc_benchmarks import RCC4

problem = RCC4(n_var=10)
```

## Interface

Each benchmark problem provides:

- `problem.n_var`: number of decision variables.
- `problem.n_obj`: number of objectives, always 2.
- `problem.xl`: lower bounds of decision variables.
- `problem.xu`: upper bounds of decision variables.
- `problem.evaluate(x)`: objective evaluation function.

The input `x` can be:

- a single decision vector with shape `(n_var,)`, or
- a batch of decision vectors with shape `(n_samples, n_var)`.

The output is always an objective matrix with shape `(n_samples, 2)`.

All problems are formulated as **bi-objective minimization** problems.

## Benchmark Problems

| Problem | Front Shape | Core Structure | Main Purpose |
|---|---|---|---|
| RCC1 | Convex | Gaussian trap + linear slope | Baseline robustness-convergence conflict |
| RCC2 | Concave | Boundary Gaussian well + interior quadratic basin | Geometry change and fragile/robust position swap |
| RCC3 | Convex | Shifted Gaussian landscape | Position-dependent robustness and variable linkage |
| RCC4 | Convex | Twin-basin landscape | Large gradient contrast between fragile and robust basins |
| RCC5 | Convex | Twin-basin landscape + non-uniform `f1` mapping | Density bias and stronger trap attraction |
| RCC6 | Convex | Coupled Gaussian trap | Structural coupling through adjacent-variable differences |

## Decision Variable Ranges

Unless otherwise stated:

```text
x_1 in [0, 1]
x_i in [-1, 1], i = 2, ..., n
```

For RCC3:

```text
x_i in [0, 1], i = 1, ..., n
```

## Noise Perturbation

The benchmark functions themselves are deterministic. For robust optimization experiments, decision-variable perturbations can be applied externally, for example:

```python
noise_ratio = 0.1
noise_amplitude = noise_ratio * (problem.xu - problem.xl)

delta = np.random.uniform(
    -noise_amplitude,
    noise_amplitude,
    size=(1000, problem.n_var),
)

x_noisy = np.clip(x[0] + delta, problem.xl, problem.xu)
f_noisy = problem.evaluate(x_noisy)
```

## Example: Evaluate All Problems

```python
import numpy as np
from rcc_benchmarks import PROBLEM_MAP

for name, problem_class in PROBLEM_MAP.items():
    problem = problem_class(n_var=10)
    x = np.random.uniform(problem.xl, problem.xu, size=(50, problem.n_var))
    f = problem.evaluate(x)
    print(name, f.shape, f[:3])
```

## Citation

If you use the RCC benchmark suite in your research, please cite the corresponding paper:

```bibtex
@article{jiang2026rcc,
  title   = {Evaluating Robust Multi-objective Optimization via USP-Based Metrics and Robustness-Convergence Conflict Benchmarks},
  author  = {Jiang, Wenxiang and Xu, Lihong},
  journal = {Under review},
  year    = {2026}
}
```

