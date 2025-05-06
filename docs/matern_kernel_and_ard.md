# Matérn Kernels and Automatic Relevance Determination

## Introduction

Kernel functions (also called covariance functions) are central to Gaussian Process (GP) models, which form the foundation of Bayesian optimization techniques like qNEHVI. The kernel function defines the similarity between data points and thereby encodes our prior beliefs about the function we're trying to model.

This document provides an in-depth explanation of Matérn kernels and Automatic Relevance Determination (ARD), which are used by default in our qNEHVI implementation.

## Matérn Kernels

### Definition

The Matérn kernel is a stationary kernel that generalizes the widely-used Radial Basis Function (RBF) kernel, providing more flexibility in modeling functions with different levels of smoothness.

The Matérn kernel between two points x and x' is defined as:

```
k(x, x') = σ² * [2^(1-ν) / Γ(ν)] * (√(2ν) * r/ℓ)^ν * K_ν(√(2ν) * r/ℓ)
```

Where:
- σ² is the output variance (amplitude)
- ν is the smoothness parameter
- Γ is the gamma function
- K_ν is the modified Bessel function of the second kind of order ν
- r = ||x - x'|| is the Euclidean distance between the points
- ℓ is the lengthscale parameter

### Smoothness Parameter (ν)

The smoothness parameter ν controls the differentiability of the resulting GP:

- As ν → ∞: The Matérn kernel approaches the RBF kernel, which is infinitely differentiable
- ν = 1/2: Equivalent to the exponential kernel, yielding non-differentiable sample paths
- ν = 3/2: Yields once-differentiable functions
- ν = 5/2: Yields twice-differentiable functions (this is the default in BoTorch)

### Common Simplifications

For half-integer values of ν, the Matérn kernel has simpler closed-form expressions:

#### Matérn 1/2 (Exponential Kernel)
```
k(x, x') = σ² * exp(-r/ℓ)
```

#### Matérn 3/2
```
k(x, x') = σ² * (1 + √(3)*r/ℓ) * exp(-√(3)*r/ℓ)
```

#### Matérn 5/2 (Default in BoTorch)
```
k(x, x') = σ² * (1 + √(5)*r/ℓ + 5*r²/(3*ℓ²)) * exp(-√(5)*r/ℓ)
```

### Properties and Advantages

1. **Flexibility**: The smoothness parameter ν allows adaptation to functions with different levels of differentiability.
2. **Generalization**: Includes the RBF kernel as a special case when ν → ∞.
3. **Realistic Modeling**: Many physical processes exhibit non-infinite differentiability, making Matérn kernels more appropriate than RBF kernels in many applications.
4. **Mean Square Differentiability**: Sample paths from a GP with a Matérn kernel are ⌊ν⌋ times differentiable in the mean square sense.

## Automatic Relevance Determination (ARD)

### Definition

Automatic Relevance Determination is a feature incorporated into kernels to automatically identify which input dimensions are most relevant for predicting the output.

Instead of using a single lengthscale parameter ℓ for all dimensions, ARD assigns a separate lengthscale parameter to each input dimension:

```
r²_ARD = Σ (x_d - x'_d)²/ℓ_d²
```

Where:
- x_d and x'_d are the dth components of input vectors x and x'
- ℓ_d is the lengthscale for dimension d

### Matérn Kernel with ARD

The Matérn kernel with ARD becomes:

```
k(x, x') = σ² * [2^(1-ν) / Γ(ν)] * (√(2ν) * √(r²_ARD))^ν * K_ν(√(2ν) * √(r²_ARD))
```

### Properties and Advantages

1. **Feature Selection**: Larger lengthscales in certain dimensions indicate less relevance, effectively performing automatic feature selection.
2. **Interpretability**: The learned lengthscale values provide insights into which parameters most affect the output.
3. **Efficiency in High Dimensions**: Helps prevent the "curse of dimensionality" by focusing on relevant dimensions.
4. **Anisotropic Modeling**: Allows the GP to model functions with different rates of change along different dimensions.

## Implementation in BoTorch and qNEHVI

In our qNEHVI implementation, we use BoTorch's `SingleTaskGP` model, which uses a Matérn 5/2 kernel with ARD by default:

```python
model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
```

### Hyperparameter Optimization

The kernel hyperparameters (lengthscales and output variance) are optimized by maximizing the marginal log-likelihood:

```python
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

### Custom Kernel Specification

If needed, we can explicitly define a custom kernel using GPyTorch:

```python
from gpytorch.kernels import MaternKernel, ScaleKernel

# Create a custom kernel with specific ν and ARD
base_kernel = MaternKernel(nu=2.5, ard_num_dims=input_dim)
kernel = ScaleKernel(base_kernel)

# Use it in a SingleTaskGP
model = SingleTaskGP(X, y, covar_module=kernel, outcome_transform=Standardize(m=1))
```

## Practical Considerations

1. **Choosing ν**: The default ν=2.5 (Matérn 5/2) is a good balance for most problems, but:
   - Use lower ν (e.g., 1/2 or 3/2) for non-smooth or highly variable functions
   - Use higher ν or RBF for very smooth functions

2. **ARD and Small Datasets**: 
   - ARD introduces more parameters (one lengthscale per dimension)
   - With small datasets, this might lead to overfitting
   - Consider using a single lengthscale (isotropic kernel) for very small datasets

3. **Prior Specification**:
   - Priors on lengthscales can be specified to incorporate domain knowledge
   - Common priors include log-normal or gamma distributions

4. **Numerical Stability**:
   - Very small or large lengthscales can cause numerical issues
   - Normalize input data to improve stability

5. **Computational Considerations**:
   - Matérn kernels are more computationally expensive than RBF kernels
   - The cost increases with the complexity of ν

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. In Advances in Neural Information Processing Systems.
3. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE, 104(1), 148-175.
4. Garnett, R. (2023). Bayesian Optimization. Cambridge University Press. 