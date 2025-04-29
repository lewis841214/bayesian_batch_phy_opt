import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import gpytorch
import torch.nn as nn
import torch.optim as optim

import botorch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from pymoo.indicators.hv import HV
import time
from tqdm import tqdm

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.botorch_adapter import BoTorchAdapter
from src.algorithms.bayesian.qnehvi import QNEHVI


# Define the neural network model for mean prediction
class MLP(nn.Module):
    """Multi-layer perceptron for mean function estimation"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1, dtype=torch.float64):
        """
        Initialize MLP network
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (typically 1)
            dtype: Data type to use for the model
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        
        # Construct layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layer = nn.Linear(prev_dim, dim, dtype=dtype)
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim, dtype=dtype))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input has the correct dtype
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)
        return self.model(x)


# Define a custom mean module for GPyTorch
class NeuralNetworkMean(gpytorch.means.Mean):
    """
    Neural network mean module for GPyTorch models.
    
    This allows the GP to use a neural network for mean prediction
    while maintaining standard GP variance/uncertainty.
    """
    
    def __init__(self, nn_model: nn.Module, target_dim=None):
        """
        Initialize with a neural network model
        
        Args:
            nn_model: PyTorch neural network model
            target_dim: Expected dimension of output (1D or 2D)
        """
        super().__init__()
        self.nn_model = nn_model
        self.target_dim = target_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network"""
        # Ensure inputs have correct dtype
        if x.dtype != next(self.nn_model.parameters()).dtype:
            x = x.to(dtype=next(self.nn_model.parameters()).dtype)
            
        # Get predictions from neural network
        pred = self.nn_model(x)
        
        # Handle batched inputs from the acquisition function
        if x.dim() == 3:
            # If model outputs 1D, expand to match targets
            if pred.dim() == 2 and self.target_dim == 1:
                return pred.squeeze(-1)
            # If model outputs 2D but targets are 1D, squeeze last dim
            elif pred.dim() == 3 and self.target_dim == 1:
                return pred.squeeze(-1)
            # Default: return as is
            return pred
        else:
            # If model outputs [n,1] but targets are [n]
            if pred.dim() == 2 and self.target_dim == 1:
                return pred.squeeze(-1)
            # If model outputs [n] but targets are [n,1]
            elif pred.dim() == 1 and self.target_dim == 2:
                return pred.unsqueeze(-1)
            # Default
            return pred


# Create custom GP model with neural network mean
class NeuralNetworkGP(SingleTaskGP):
    """
    Gaussian Process with neural network mean function.
    
    This hybrid model uses a neural network for mean prediction
    while maintaining GP-based uncertainty estimates.
    """
    
    def __init__(
        self, 
        train_X: torch.Tensor, 
        train_Y: torch.Tensor, 
        nn_model: nn.Module,
        likelihood=None,
        outcome_transform=None
    ):
        """
        Initialize the neural network GP model
        
        Args:
            train_X: Training inputs
            train_Y: Training targets
            nn_model: Neural network model for mean prediction
            likelihood: GPyTorch likelihood
            outcome_transform: Outcome transform
        """
        # Print input shapes for debugging
        print(f"NeuralNetworkGP init: train_X shape={train_X.shape}, train_Y shape={train_Y.shape}")
        
        # Store original Y dimension for the mean module
        target_dim = train_Y.dim()
        
        # Create a default likelihood if none provided
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            
        # Initialize with custom mean module that knows the target dimension
        mean_module = NeuralNetworkMean(nn_model, target_dim=target_dim)
        
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        # Initialize the parent GP model with our custom mean
        super().__init__(
            train_X=train_X, 
            train_Y=train_Y, 
            likelihood=likelihood,
            mean_module=mean_module,
            outcome_transform=outcome_transform
        )
        
        # Store neural network model
        self.nn_model = nn_model


class NNQNEHVI(QNEHVI):
    """
    Neural Network enhanced q-Noisy Expected Hypervolume Improvement (qNEHVI)
    
    This implementation uses neural networks for mean prediction
    while maintaining GP-based uncertainty estimates.
    
    References:
    [1] S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of 
        Multiple Noisy Objectives with Expected Hypervolume Improvement. Advances 
        in Neural Information Processing Systems 34, 2021.
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
        nn_epochs: int = 300,
        nn_batch_size: int = 16,
        nn_regularization: float = 1e-4,
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
            nn_batch_size: Batch size for neural network training
            nn_regularization: L2 regularization for neural network
        """
        # Store neural network parameters for future enhancements
        self.nn_layers = nn_layers
        self.nn_learning_rate = nn_learning_rate
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.nn_regularization = nn_regularization
        
        # Track model training metrics
        self.model_metrics = {
            'train_losses': [],
            'val_losses': []
        }
        
        # Initialize neural network models
        self.nn_models = []
        
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
    
    def _train_neural_network(self, X: torch.Tensor, y: torch.Tensor, input_dim: int, objective_idx: int) -> nn.Module:
        """
        Train neural network for mean prediction
        
        Args:
            X: Training inputs
            y: Training targets
            input_dim: Input dimension
            objective_idx: Index of the objective this network predicts
            
        Returns:
            Trained neural network model
        """
        print(f"Training neural network for objective {objective_idx+1}...")
        
        # Get the data type from input tensor for consistency
        dtype = X.dtype
        print(f"Using dtype: {dtype}")
        
        # Create neural network model with the same dtype as the input data
        model = MLP(input_dim=input_dim, hidden_dims=self.nn_layers, dtype=dtype)
        
        # Use MSE loss and Adam optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.nn_learning_rate, 
            weight_decay=self.nn_regularization
        )
        
        # Split data into training and validation sets (80/20)
        n_train = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Create data loader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=min(self.nn_batch_size, len(X_train)),
            shuffle=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        # Use tqdm for progress tracking
        for epoch in tqdm(range(self.nn_epochs), desc=f"NN training (obj {objective_idx+1})"):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.nn_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Store training metrics
        if len(self.model_metrics['train_losses']) <= objective_idx:
            self.model_metrics['train_losses'].append(train_losses)
            self.model_metrics['val_losses'].append(val_losses)
        else:
            self.model_metrics['train_losses'][objective_idx] = train_losses
            self.model_metrics['val_losses'][objective_idx] = val_losses
            
        print(f"Neural network training completed. Final val loss: {best_val_loss:.6f}")
        
        return model
        
    def _update_model(self):
        """Update the hybrid model with current training data"""
        start_time = time.time()
        
        if len(self.train_x) < 2:
            # Not enough data to fit a model
            return None
            
        # Normalize inputs
        bounds_t = self.bounds.transpose(0, 1)  # Make it (n_dims, 2)
        X = normalize(self.train_x, bounds_t)
        Y = self.train_y
        
        print(f"Fitting hybrid NN-GP model with {len(X)} observations...")
        print(f"Input tensor shapes: X={X.shape}, Y={Y.shape}")
        
        # Create and fit a model for each objective
        models = []
        self.nn_models = []  # Reset neural network models
        
        for i in tqdm(range(self.n_objectives), desc="Fitting hybrid models"):
            y = Y[:, i]  # Get ith objective, keep as 1D
            print(f"Objective {i+1}: y shape = {y.shape}, y dtype = {y.dtype}")
            
            # Train neural network for mean prediction
            input_dim = X.shape[1]
            nn_model = self._train_neural_network(X, y.unsqueeze(-1), input_dim, i)  # Pass 2D for training
            self.nn_models.append(nn_model)
            
            # Create GP model with neural network mean
            model = NeuralNetworkGP(
                train_X=X,
                train_Y=y,  # Pass 1D tensor as BoTorch expects
                nn_model=nn_model,
                outcome_transform=Standardize(m=1)
            )
            
            # Fit GP parameters (keeping neural network fixed)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            # Only optimize GP hyperparameters (not neural network)
            fit_gpytorch_mll(mll)
            
            # Test mean function to ensure dimensions match
            with torch.no_grad():
                test_mean = model.mean_module(X)
                print(f"Mean function output shape: {test_mean.shape}, targets shape: {y.shape}")
            
            models.append(model)
        
        # Create a ModelListGP from the individual models
        model_list = ModelListGP(*models)
            
        elapsed = time.time() - start_time
        self.timing_history['model_update'].append(elapsed)
        print(f"Model fitting completed in {elapsed:.2f} seconds")
        
        return model_list
    
    def get_model_metrics(self):
        """Return neural network training metrics"""
        return self.model_metrics 

    def ask(self):
        """Return a batch of points to evaluate"""
        # Calculate remaining budget and effective batch size
        remaining = self.budget - self.evaluated_points
        effective_batch_size = min(self.batch_size, remaining)
        
        if effective_batch_size <= 0:
            print("Budget exhausted, no more evaluations possible.")
            return []
            
        print(f"Generating batch of {effective_batch_size} candidates. Remaining budget: {remaining}")
        
        # If we don't have enough points, generate random samples
        if len(self.train_x) < 2 * self.n_objectives:
            print("Insufficient data for GP model. Using random sampling.")
            
            # Generate candidates directly using the parameter space
            candidates = []
            for _ in range(effective_batch_size):
                random_params = {}
                # Sample each parameter type appropriately
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        # Sample continuous uniform
                        random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        # Sample integer uniform
                        random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        # Sample categorical uniform
                        if 'categories' in config:
                            random_params[name] = np.random.choice(config['categories'])
                        else:
                            random_params[name] = np.random.choice(config['values'])
                candidates.append(random_params)
                
            return candidates
        
        # Update GP models
        model_list = self._update_model()
        
        if model_list is None:
            # Fall back to random sampling
            # Generate candidates directly using the parameter space
            candidates = []
            for _ in range(effective_batch_size):
                random_params = {}
                # Sample each parameter type appropriately
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        if 'categories' in config:
                            random_params[name] = np.random.choice(config['categories'])
                        else:
                            random_params[name] = np.random.choice(config['values'])
                candidates.append(random_params)
                
            return candidates
        
        start_time = time.time()
        
        # Create the acquisition function
        print("Creating qNEHVI acquisition function...")
        
        # Get current Pareto front
        Y = self.train_y
        try:
            # Use proper tensor cloning
            partitioning = FastNondominatedPartitioning(ref_point=self.ref_point.clone().detach())
            partitioning.update(Y)
            pareto_Y = partitioning.pareto_Y
        except Exception as e:
            print(f"Error in partitioning: {e}. Using all observed points.")
            pareto_Y = Y
        
        # Normalize inputs for acquisition function
        bounds_t = self.bounds.transpose(0, 1)
        X_baseline = normalize(self.train_x, bounds_t)
        
        # Create acquisition function
        try:
            # Create a sampler with the specified number of MC samples
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
            
            # Ensure alpha is a valid value (default to 1e-3 if None)
            alpha = 1e-3 if self.noise_std is None else self.noise_std
            
            # Debug print data types
            print(f"DEBUG - ref_point dtype: {self.ref_point.dtype}, X_baseline dtype: {X_baseline.dtype}")
            
            # Create the acquisition function with the sampler - use qLogNoisyExpectedHypervolumeImprovement instead
            try:
                # Try to use the improved LogNEHVI implementation if available
                from botorch.acquisition.multi_objective.monte_carlo import qLogNoisyExpectedHypervolumeImprovement
                acq_func = qLogNoisyExpectedHypervolumeImprovement(
                    model=model_list,
                    ref_point=self.ref_point.clone().detach(),
                    X_baseline=X_baseline,
                    prune_baseline=True,
                    alpha=alpha,
                    sampler=sampler
                )
                print("Using qLogNoisyExpectedHypervolumeImprovement")
            except ImportError:
                # Fall back to original NEHVI if the Log version isn't available
                acq_func = qNoisyExpectedHypervolumeImprovement(
                    model=model_list,
                    ref_point=self.ref_point.clone().detach(),
                    X_baseline=X_baseline,
                    prune_baseline=True,
                    alpha=alpha,
                    sampler=sampler
                )
                print("Using qNoisyExpectedHypervolumeImprovement")
                
        except Exception as e:
            print(f"Error creating acquisition function: {e}")
            # Fall back to random sampling
            candidates = []
            for _ in range(effective_batch_size):
                random_params = {}
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        if 'categories' in config:
                            random_params[name] = np.random.choice(config['categories'])
                        else:
                            random_params[name] = np.random.choice(config['values'])
                candidates.append(random_params)
            return candidates
        
        # Optimize acquisition function
        print(f"Optimizing acquisition function to find {effective_batch_size} candidates...")
        
        # Use standard bounds [0, 1] for optimization
        standard_bounds = torch.zeros(2, X_baseline.shape[1], dtype=torch.double)
        standard_bounds[1] = 1.0
        
        # Reduce number of restarts and raw samples for faster optimization
        n_restarts = 1
        raw_samples = 50
        
        try:
            # Debug print for optimization settings
            print(f"DEBUG - Optimization settings: batch_size={effective_batch_size}, num_restarts={n_restarts}, raw_samples={raw_samples}")
            
            # Try with sequential=True first for better numerical stability
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=effective_batch_size,
                num_restarts=n_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 100, "ftol": 1e-5, "method": "L-BFGS-B"},
                sequential=True,
            )
            
            if acq_values.numel() > 1:  # If we get multiple values
                print(f"Acquisition values shape: {acq_values.shape}, taking mean")
                acq_value_scalar = acq_values.mean().item()
            else:
                acq_value_scalar = acq_values.item()
                
            print(f"Acquisition value: {acq_value_scalar:.6f}")
            
            # Unnormalize candidates
            candidates = unnormalize(candidates.detach(), bounds_t)
            
        except Exception as e:
            print(f"Error in acquisition optimization: {e}. Using random samples.")
            # Generate random samples instead
            candidates = []
            for _ in range(effective_batch_size):
                random_params = {}
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        if 'categories' in config:
                            random_params[name] = np.random.choice(config['categories'])
                        else:
                            random_params[name] = np.random.choice(config['values'])
                candidates.append(random_params)
            
            # Convert to tensor format
            candidate_dicts = candidates
            return candidate_dicts
        
        elapsed = time.time() - start_time
        self.timing_history['acquisition_optimization'].append(elapsed)
        print(f"Acquisition optimization completed in {elapsed:.2f} seconds")
        
        # Convert tensor to dictionaries
        candidate_dicts = self._tensors_to_dicts(candidates)
        
        # Return a batch of candidates (ensuring we don't exceed batch size)
        return candidate_dicts[:effective_batch_size] 