# Multi-Objective Optimization Algorithms Implementation

## Summary of Algorithms

| Algorithm | Type | Key Features | Parallelization Approach | Non-batched Off-the-shelf | Batch Implementation Approach |
|-----------|------|-------------|--------------------------|---------------------------|-------------------------------|
| q-ParEGO | Bayesian Optimization | Scalarizes multiple objectives with random weights | Batch acquisition with q-EI | True (BoTorch, PyGMO) | Greedy selection with hallucinated observations using posterior mean |
| q-EHVI | Bayesian Optimization | Directly maximizes Pareto hypervolume | Batch expected hypervolume improvement | True (BoTorch) | Monte Carlo sampling of joint hypervolume improvement across batch |
| BUCB | Bayesian Optimization | Upper confidence bound with hallucinated observations | Sequential selection with diversity | True (GPyTorch, BoTorch) | Sequential greedy selection with GP model updates after each point |
| NSGA-III | Evolutionary Algorithm | Reference points in objective space | Population-based parallel evaluation | True (pymoo) | Natural batch parallelism through population evaluation |
| MOEA/D | Evolutionary Algorithm | Decomposes into subproblems | Simultaneous subproblem evaluation | True (pymoo) | Parallel evaluation of decomposed subproblems |
| MOACO | Ant Colony Optimization | Multiple pheromone matrices | Batch parallel ant dispatching | False | Parallel ant dispatch with synchronized pheromone updates |
| PABO | Physics-Aware BO | Hierarchical prediction through physical representations | Transformer-based parallel surrogate | False | Transformer-based acquisition with diversity preservation |

This document outlines the algorithms we plan to implement for multi-objective optimization in physical design parameter tuning, leveraging the [nevergrad](https://github.com/facebookresearch/nevergrad) framework.

## 1. Bayesian Optimization with Batch Acquisition

### 1.1. q-ParEGO (q-Pareto Efficient Global Optimization)
- **Description**: Scalarizes multiple objectives using random weights and optimizes batch of points using q-EI (q-Expected Improvement)
- **Mathematical Expression**:
  ```
  α_q-ParEGO(x^(1), ..., x^(b)) = E[max_{i=1}^b max(0, s(y*) - s(f(x^(i))))]
  ```
  where s is a scalarization function with random weights
- **Implementation Focus**: 
  - Implement scalarization with random weights
  - Extend to batch setting with q-EI
  - Integrate with nevergrad's optimization framework

### 1.2. q-EHVI (q-Expected Hypervolume Improvement)
- **Description**: Directly maximizes expected improvement in Pareto hypervolume, accounting for interactions between batch points
- **Mathematical Expression**:
  ```
  α_q-EHVI(x^(1), ..., x^(b)) = E[HV(P_t ∪ {f(x^(1)), ..., f(x^(b))}) - HV(P_t)]
  ```
  where P_t is the current Pareto front and HV is the hypervolume indicator
- **Implementation Focus**:
  - Calculate hypervolume indicator efficiently
  - Monte Carlo approximation of expected improvement
  - Parallel evaluation of hypervolume contributions

### 1.3. BUCB (Batch Upper Confidence Bound)
- **Description**: Extends UCB acquisition to batch setting with hallucinated observations for information gain from pending evaluations
- **Mathematical Expression** for selecting the batch sequentially:
  ```
  x_{t,j} = argmax_{x ∈ X} μ_{t,j-1}(x) + β_t^{1/2}σ_{t,j-1}(x)
  ```
  where μ_{t,j-1} and σ_{t,j-1} are the posterior mean and standard deviation after adding j-1 fantasy points
- **Implementation Focus**:
  - GP model updating with hallucinated observations
  - Sequential batch point selection
  - Diversity promotion mechanism

## 2. Evolutionary Algorithms for Batch Pareto Optimization

### 2.1. NSGA-III with Batch Parallelism
- **Description**: Employs reference points in objective space to maintain diversity with natural parallelism
- **Key Components**:
  - Selection based on dominance ranking and reference-point-based niching
  - Reference points on a normalized hyperplane
- **Implementation Focus**:
  - Leverage nevergrad's population-based optimization capabilities
  - Implement reference point generation mechanism
  - Integrate batch evaluation capabilities

### 2.2. MOEA/D (Multi-Objective EA based on Decomposition)
- **Description**: Decomposes problem into subproblems solved simultaneously
- **Mathematical Approach**:
  ```
  g^te(x|λ, z*) = max_{1 ≤ i ≤ m} {λ_i|f_i(x) - z_i*|}
  ```
  where λ is a weight vector and z* is the reference point
- **Implementation Focus**:
  - Create weight vector generation mechanism
  - Implement decomposition approaches (Tchebycheff, weighted sum, etc.)
  - Integrate neighborhood-based mating restrictions

## 3. Ant Colony Optimization for Batch Pareto Frontier

### 3.1. Multi-Objective Ant Colony Optimization (MOACO)
- **Description**: Maintains multiple pheromone matrices (one per objective) with batch parallel dispatching
- **Pheromone Update Rule**:
  ```
  τ_{ij}^k ← (1-ρ)τ_{ij}^k + Σ_{a=1}^b Δτ_{ij}^{k,a}
  ```
  where Δτ_{ij}^{k,a} is the pheromone deposit by ant a for objective k
- **Implementation Focus**:
  - Create multiple pheromone matrix management
  - Implement batch parallel ant dispatching
  - Develop Pareto-based pheromone deposit strategies

## 4. Physics-Aware Bayesian Optimization (PABO)

### 4.1. Transformer-Based Surrogate Model
- **Description**: Neural network surrogate that models relationships between parameters, intermediate physical representations, and final PPA metrics
- **Key Components**:
  - Parameter encoder using transformer architecture
  - 2D grid generation through vision transformer
  - Cross-modal fusion for final PPA prediction
- **Implementation Focus**:
  - Develop transformer architecture within nevergrad framework
  - Create mechanisms for handling mixed parameter types
  - Implement hierarchical prediction pipeline

## 5. Parameter Space Handling Across Frameworks

When implementing optimization algorithms, the parameter space definition is a critical component. Below is a comparison of how different optimization frameworks handle parameter spaces, which will inform our implementation strategy.

### Key Differences Summary

| Framework | Continuous | Integer | Categorical | Mixed Space | Constraints |
|-----------|------------|---------|-------------|-------------|-------------|
| Nevergrad | `ng.p.Scalar` | `Scalar().set_integer_casting()` | `ng.p.Choice` | `ng.p.Instrumentation` | Built-in handling |
| BoTorch | Tensor bounds | Manual conversion | Manual one-hot encoding | Manual implementation | Custom acquisition functions |
| Pymoo | Problem definition | Problem definition with vtype | Via integer mapping | Native support | Native support |
| Optuna | `suggest_float` | `suggest_int` | `suggest_categorical` | Native support | Via constraints in objective |
| PyGMO | Continuous bounds | get_nix() method | Manual encoding | Partial support | Multiple approaches |

### Framework Comparison Notes

1. **Nevergrad**
   - Strengths: Clean API for mixed parameter types, instrumentation for complex spaces
   - Limitations: Less specialized for certain algorithm types

2. **BoTorch**
   - Strengths: Powerful Bayesian optimization capabilities, PyTorch integration
   - Limitations: Requires manual handling of discrete/categorical parameters

3. **Pymoo**
   - Strengths: Excellent for evolutionary algorithms, native mixed-variable support
   - Limitations: Less suited for Bayesian optimization approaches

4. **Integration Considerations**
   - Converting between parameter space representations requires careful attention
   - Wrapper classes may be needed to maintain consistent experiment interface
   - Categorical parameter handling differs most significantly between frameworks

### Parameter Space Example

```python
# Nevergrad example for a mixed parameter space
import nevergrad as ng

param_space = ng.p.Instrumentation(
    continuous_param=ng.p.Scalar(lower=0.0, upper=10.0),
    integer_param=ng.p.Scalar(lower=1, upper=100).set_integer_casting(),
    categorical_param=ng.p.Choice(["option1", "option2", "option3"]),
    log_param=ng.p.Log(lower=0.001, upper=1.0)
)
```

## Implementation Strategy

For each algorithm, we will:

1. Create a dedicated implementation class derived from nevergrad's optimization primitives
2. Develop unit tests to verify correctness and performance
3. Benchmark against standard test functions and physical design examples
4. Document usage examples and integration patterns

## Timeline

1. **Phase 1** (Weeks 1-2): Basic implementations of batch Bayesian optimization methods
2. **Phase 2** (Weeks 3-4): Evolutionary and ant colony optimization implementations
3. **Phase 3** (Weeks 5-6): Physics-aware BO implementation
4. **Phase 4** (Weeks 7-8): Testing, benchmarking, and documentation

## References

1. BO_physical.md (internal document)
2. [Nevergrad GitHub Repository](https://github.com/facebookresearch/nevergrad)
3. Desautels et al. (2014), "Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization" 