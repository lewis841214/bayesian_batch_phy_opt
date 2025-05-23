# Bayesian Batch Physics-Aware Optimization

A framework-agnostic library for multi-objective optimization algorithms with batch evaluation capabilities. The library focuses on Bayesian optimization, evolutionary algorithms, and physics-aware approaches.

## Features

- Framework-agnostic parameter space definition
- Adapters for popular optimization frameworks like BoTorch, PyMOO, PyGMO, etc.
- Batch parallel evaluation for all algorithms
- Multi-objective optimization with direct Pareto front optimization
- Physics-aware surrogate models for specialized applications

## Algorithms

| Algorithm | Type | Key Features | Batch Approach |
|-----------|------|-------------|----------------|
| q-ParEGO | Bayesian Optimization | Scalarizes objectives with random weights | Greedy selection with hallucinated observations |
| q-EHVI | Bayesian Optimization | Directly maximizes Pareto hypervolume | Monte Carlo sampling of joint hypervolume improvement |
| BUCB | Bayesian Optimization | Upper confidence bound with hallucination | Sequential greedy selection with model updates |
| NSGA-III | Evolutionary Algorithm | Reference points in objective space | Population-based parallel evaluation |
| MOEA/D | Evolutionary Algorithm | Decomposes into subproblems | Parallel evaluation of subproblems |
| MOACO | Ant Colony Optimization | Multiple pheromone matrices | Batch parallel ant dispatching |
| PABO | Physics-Aware BO | Hierarchical prediction through physical representations | Transformer-based parallel surrogate |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bayesian_batch_phy_opt.git
cd bayesian_batch_phy_opt

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

``` # to test multiple problems with multiple optimization algorithm

python tests/test_algorithms/unified_benchmark.py --compare --algorithms qnehvi qehvi qparego nsga2 moead nsga3 --problems nonlinear constrained --budget 20 --batch-size 5 --output-dir output/multi_problem_benchmark
```

``` only test one 

python tests/test_algorithms/unified_benchmark.py --compare --algorithms nsga2 moead nsga3 qnehvi qehvi qparego --problem complex_categorical --budget 500 --batch-size 20 --output-dir output/complex_categorical_comparison_all


python tests/test_algorithms/unified_benchmark.py --algorithm nsga2 --problem category_matrix --budget 50 --batch-size 10 --output-dir output/category_matrix_nsga2
```

``` # test a batch
 python tests/test_algorithms/unified_benchmark.py --compare --algorithms nsga2 moead nsga3 qnehvi nn-qnehvi xgb-qnehvi   --problem complex_categorical --budget 500 --batch-size 20 --output-dir output/complex_categorical_comparison_all

```

```# test all problems, all algorithm
python tests/test_algorithms/batch_benchmark.py --budget 50 --batch-size 10 --output-dir output/full_comparison_50_10
```
Here's a simple example of using the q-EHVI algorithm on a multi-objective problem:

```python
from src.core.parameter_space import ParameterSpace
from src.algorithms.bayesian.q_ehvi import QEHVI

# Define parameter space
space = ParameterSpace()
space.add_continuous_parameter('x1', 0.0, 1.0)
space.add_continuous_parameter('x2', 0.0, 1.0)
space.add_integer_parameter('x3', 1, 10)

# Define objective function (must return a list of objective values)
def objective_function(params):
    x1, x2, x3 = params['x1'], params['x2'], params['x3']
    f1 = x1**2 + x2**2 + x3
    f2 = (x1-1)**2 + (x2-1)**2 + x3
    return [f1, f2]  # Minimize both objectives

# Create optimizer
optimizer = QEHVI(
    parameter_space=space,
    budget=50,            # Total evaluation budget
    batch_size=5,         # Evaluate 5 points in parallel
    n_objectives=2,
    reference_point=[100, 100]  # For hypervolume calculation
)

# Run optimization
result = optimizer.minimize(objective_function)

# Get Pareto front
pareto_points, pareto_values = result
```

## Project Structure

```
bayesian_batch_phy_opt/
├── docs/                           # Documentation
├── src/                            # Source code
│   ├── core/                       # Core framework
│   │   ├── parameter_space.py      # Framework-agnostic parameter space
│   │   ├── framework_adapter.py    # Framework adapter base class
│   │   ├── algorithm.py            # Base optimizer class
│   │   ├── problem.py              # Problem definition
│   │   ├── metrics.py              # Performance metrics
│   │   └── __init__.py             # Package initialization
│   ├── algorithms/                 # Algorithm implementations
│   │   ├── bayesian/               # Bayesian optimization algorithms
│   │   ├── evolutionary/           # Evolutionary algorithms
│   │   ├── ant_colony/             # Ant colony optimization
│   │   └── pabo/                   # Physics-aware Bayesian optimization
│   └── adapters/                   # Framework adapters
├── tests/                          # Test suite
│   ├── test_core/                  # Tests for core functionality
│   ├── test_algorithms/            # Tests for optimization algorithms
│   │   └── test_problems.py        # Benchmark problem definitions
│   └── test_adapters/              # Tests for framework adapters
├── examples/                       # Usage examples
├── output/                         # Output directory for benchmarks
├── notebook/                       # Jupyter notebooks
├── setup.py                        # Package installation script
├── requirements.txt                # Required dependencies
├── run_examples.py                 # Script to run examples
└── run_tests.py                    # Script to run tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 