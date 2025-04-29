# Neural Network Enhanced QNEHVI Bayesian Optimization

## Overview

This document outlines our plan to enhance the existing QNEHVI (q-Noisy Expected Hypervolume Improvement) Bayesian Optimization algorithm by implementing a neural network-based surrogate model that replaces the standard GP mean function while retaining kernel-based covariance estimation for uncertainty quantification.

## Motivation

Standard Gaussian Process models in Bayesian optimization have limitations:

1. **Limited Expressiveness**: Standard kernels may not capture complex, non-stationary relationships in the parameter space
2. **Challenges with Categorical Parameters**: Traditional GPs struggle with categorical parameters that lack natural distance metrics
3. **Scaling Issues**: Performance can degrade in higher-dimensional spaces with limited samples

Neural networks offer a promising alternative for the mean function while maintaining uncertainty quantification through GP covariance structures.

## Implementation Strategy

### 1. Core Components

#### NeuralNetworkGP Model

```python
class NeuralNetworkGP(SingleTaskGP):
    """GP model that uses a neural network for mean function and standard kernel for covariance"""
    
    def __init__(
        self, 
        train_X, 
        train_Y, 
        nn_layers=[64, 32], 
        outcome_transform=None,
        **kwargs
    ):
        # Initialize the standard GP
        super().__init__(train_X, train_Y, outcome_transform=outcome_transform)
        
        # Create a neural network for the mean function
        input_dim = train_X.shape[-1]
        output_dim = train_Y.shape[-1]
        
        self.nn_mean = self._build_nn_mean(input_dim, output_dim, nn_layers)
        
    def _build_nn_mean(self, input_dim, output_dim, hidden_layers):
        """Build neural network for mean function"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # Get mean from neural network
        nn_mean = self.nn_mean(x)
        
        # Get standard GP posterior for covariance
        standard_posterior = super().forward(x)
        
        # Create new posterior with NN mean and GP covariance
        return gpytorch.distributions.MultivariateNormal(
            nn_mean, 
            standard_posterior.covariance_matrix
        )
```

#### NNQNEHVI Algorithm

```python
class NNQNEHVI(QNEHVI):
    """
    Neural Network enhanced q-Noisy Expected Hypervolume Improvement (qNEHVI)
    
    Uses neural networks for mean prediction in GPs while maintaining
    standard kernel methods for variance estimation.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        budget: int,
        batch_size: int = 1, 
        n_objectives: int = 2,
        ref_point: Optional[List[float]] = None,
        noise_std: Optional[List[float]] = None,
        mc_samples: int = 128,
        nn_layers: List[int] = [64, 32],
        nn_learning_rate: float = 0.01,
        nn_epochs: int = 100,
        **kwargs
    ):
        """
        Initialize the Neural Network enhanced qNEHVI optimizer
        
        Args:
            parameter_space: Parameter space to optimize
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            n_objectives: Number of objectives to optimize
            ref_point: Reference point for hypervolume calculation
            noise_std: Standard deviation of observation noise for each objective
            mc_samples: Number of MC samples for acquisition function approximation
            nn_layers: Hidden layer sizes for the neural network mean function
            nn_learning_rate: Learning rate for neural network training
            nn_epochs: Number of epochs for neural network training
        """
        self.nn_layers = nn_layers
        self.nn_learning_rate = nn_learning_rate
        self.nn_epochs = nn_epochs
        
        super().__init__(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=n_objectives,
            ref_point=ref_point,
            noise_std=noise_std,
            mc_samples=mc_samples,
            **kwargs
        )
```

#### Custom Training Loop

```python
def train_neural_network_gp(model, train_X, train_Y, learning_rate=0.01, epochs=100):
    """Custom training loop for neural network GP"""
    # Set up optimizers - one for GP parameters, one for NN parameters
    gp_optimizer = torch.optim.Adam([
        {'params': model.covar_module.parameters()},
        {'params': model.likelihood.parameters()}
    ], lr=learning_rate)
    
    nn_optimizer = torch.optim.Adam(model.nn_mean.parameters(), lr=learning_rate)
    
    # Set up loss function
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    # Training loop
    for epoch in range(epochs):
        # Zero gradients
        gp_optimizer.zero_grad()
        nn_optimizer.zero_grad()
        
        # Forward pass
        output = model(train_X)
        
        # Calc loss
        loss = -mll(output, train_Y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        gp_optimizer.step()
        nn_optimizer.step()
```

### 2. Adapter Implementation

```python
class QNEHVIHybridAdapter(BayesianOptAdapter):
    """
    Adapter for Neural Network enhanced qNEHVI Bayesian optimization
    
    This adapter handles the integration of neural network surrogate models
    with the standard qNEHVI algorithm for multi-objective optimization.
    """
    
    def __init__(self, surrogate_model="nn", nn_config=None):
        """
        Initialize adapter with neural network configuration
        
        Args:
            surrogate_model: Type of surrogate model to use ('nn' or 'xgboost')
            nn_config: Optional configuration for neural network
        """
        self.surrogate_model = surrogate_model
        self.nn_config = nn_config or {
            'hidden_layers': [64, 32],
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 16,
            'regularization': 1e-4
        }
        
        # Initialize with NNQNEHVI algorithm class
        super().__init__(NNQNEHVI)
    
    def setup(self, problem, budget, batch_size):
        """Setup Neural Network enhanced Bayesian optimizer"""
        parameter_space = problem.get_parameter_space()
        ref_point = problem.get_reference_point()
        
        # Initialize algorithm with neural network configuration
        self.algorithm = NNQNEHVI(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128,
            
            # Neural network specific parameters
            nn_layers=self.nn_config['hidden_layers'],
            nn_learning_rate=self.nn_config['learning_rate'],
            nn_epochs=self.nn_config['epochs'],
            nn_batch_size=self.nn_config['batch_size'],
            nn_regularization=self.nn_config['regularization'],
        )
        
        return self
```

## Key Implementation Details

### 1. Modified `_update_model()` Method

```python
def _update_model(self):
    """Update the GP model with current training data"""
    start_time = time.time()
    
    if len(self.train_x) < 2:
        # Not enough data to fit a GP
        return None
        
    # Normalize inputs
    bounds_t = self.bounds.transpose(0, 1)
    X = normalize(self.train_x, bounds_t)
    Y = self.train_y
    
    print(f"Fitting Neural Network GP model with {len(X)} observations...")
    
    # Create and fit a model for each objective
    models = []
    
    for i in tqdm(range(self.n_objectives), desc="Fitting NN-GP models"):
        y = Y[:, i:i+1]  # Get ith objective, keep dimension
        
        # Create neural network GP
        model = NeuralNetworkGP(X, y, nn_layers=self.nn_layers)
        
        # Train the model with custom training loop
        train_neural_network_gp(
            model, X, y, 
            learning_rate=self.nn_learning_rate, 
            epochs=self.nn_epochs
        )
        
        models.append(model)
        
    # Create a ModelListGP from the individual models
    model_list = ModelListGP(*models)
        
    elapsed = time.time() - start_time
    self.timing_history['model_update'].append(elapsed)
    print(f"Model fitting completed in {elapsed:.2f} seconds")
    
    return model_list
```

## Expected Benefits

1. **Improved Performance**
   - Better prediction accuracy for complex response surfaces
   - More efficient exploration of parameter space
   - Better handling of categorical parameters

2. **Enhanced Flexibility**
   - Ability to capture non-stationary relationships
   - Better scaling to higher-dimensional spaces
   - Improved modeling of discrete and mixed spaces

3. **Practical Advantages**
   - Maintained uncertainty quantification from GP framework
   - Compatible with existing optimization workflows
   - Configurable for different problem requirements

## Implementation Steps

1. **Create the NeuralNetworkGP class**
   - Extend SingleTaskGP with neural network mean function
   - Implement forward pass that combines NN mean with GP covariance

2. **Develop custom training procedure**
   - Handle both GP parameters and NN parameters
   - Use separate optimizers for GP and NN components
   - Maintain marginal log likelihood as objective

3. **Modify model initialization**
   - Add hyperparameters for NN architecture
   - Create constructor for NN layers

4. **Update model fitting process**
   - Replace standard GP fitting with NN-GP fitting
   - Implement early stopping or learning rate scheduling

5. **Test and validate**
   - Compare performance against standard GP
   - Tune NN architecture for best performance
   - Analyze convergence properties

## Usage Example

```python
from src.adapters.algorithm_adapters import get_algorithm_adapter

# Example usage
optimizer = get_algorithm_adapter("nn-qnehvi")
optimizer.setup(problem, budget=100, batch_size=4)

# Custom NN configuration
optimizer = get_algorithm_adapter("nn-qnehvi")
nn_config = {
    'hidden_layers': [128, 64, 32],
    'learning_rate': 0.005,
    'epochs': 200
}
optimizer.nn_config = nn_config
optimizer.setup(problem, budget=100, batch_size=4)
```

## Advanced Features to Consider

1. **Transfer Learning**: Initialize NN weights from previous runs
2. **Uncertainty Quantification**: Ensure NN-GP provides reliable uncertainty estimates
3. **Hyperparameter Tuning**: Automatically select NN architecture
4. **Hybrid Models**: Combine deep kernel learning with NN mean functions
5. **Memory Management**: Handle large batch sizes efficiently

## Performance Considerations

- Ensure the NN training doesn't significantly slow down the optimization loop
- Consider using GPU acceleration for NN training
- Implement regularization to prevent overfitting with small datasets
- Implement warm-starting for sequential optimization 