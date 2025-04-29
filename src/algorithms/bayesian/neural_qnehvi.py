import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import gpytorch
import torch.nn as nn
import torch.optim as optim
import os
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
    
    def __init__(self, input_dim: int, hidden_dims: List[int], hidden_map_dim: Optional[List[int]] = None, output_dim: int = 1, dtype=torch.float64):
        """
        Initialize MLP network
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            hidden_map_dim: Dimensions for the hidden map output (if not None)
            output_dim: Output dimension (typically 1)
            dtype: Data type to use for the model
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_map_dim = hidden_map_dim
        self.dtype = dtype
        
        # Construct layers for main network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layer = nn.Linear(prev_dim, dim, dtype=dtype)
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Final output layer for main prediction
        self.first_half = nn.Sequential(*layers)
        self.second_half = nn.Linear(prev_dim, output_dim, dtype=dtype)
        
        # Hidden map prediction network if specified
        if self.hidden_map_dim is not None:
            # Create layers for hidden map with dimensions [8, 12]
            self.map_layer = nn.Linear(prev_dim, self.hidden_map_dim[0] * self.hidden_map_dim[1], dtype=dtype)
            
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        # Ensure input has the correct dtype
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)
        
        # Forward through shared layers
        features = self.first_half(x)
        
        # Main output prediction
        output = self.second_half(features)
        
        # If hidden map is requested, generate it
        if self.hidden_map_dim is not None:
            # Generate flattened map
            flat_map = self.map_layer(features)
            
            # Reshape to proper dimensions [batch_size, 8, 12]
            batch_size = x.shape[0]
            hidden_map = flat_map.view(batch_size, self.hidden_map_dim[0], self.hidden_map_dim[1])
            
            # Return both outputs
            return output, hidden_map
        else:
            # Return only main output
            return output


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
        if type(pred) == tuple:
            pred, hidden_map = pred
        
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
        hidden_map_dim: Optional[List[int]] = None,
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
        self.hidden_map_dim = hidden_map_dim

        # Initialize hidden maps container
        self.train_hidden_maps = None

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
    
    def _train_neural_network(self, X: torch.Tensor, y: torch.Tensor, input_dim: int, objective_idx: int, hidden_maps: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Train neural network for mean prediction
        
        Args:
            X: Training inputs
            y: Training targets
            input_dim: Input dimension
            objective_idx: Index of the objective this network predicts
            hidden_maps: Target hidden maps to predict (if not None)
            
        Returns:
            Trained neural network model
        """
        print(f"Training neural network for objective {objective_idx+1}...")
        
        # Get the data type from input tensor for consistency
        dtype = X.dtype
        print(f"Using dtype: {dtype}")
        
        # Create neural network model with the same dtype as the input data
        model = MLP(
            input_dim=input_dim, 
            hidden_dims=self.nn_layers, 
            hidden_map_dim=self.hidden_map_dim if hidden_maps is not None else None,
            dtype=dtype
        )
        
        # Use combined loss: MSE for main prediction plus L2 norm for hidden map
        mse_criterion = nn.MSELoss()
        
        # Define optimizer
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
        
        # Split hidden maps if provided
        hidden_maps_train = None
        hidden_maps_val = None
        if hidden_maps is not None:
            hidden_maps_train = hidden_maps[train_indices]
            hidden_maps_val = hidden_maps[val_indices]
        
        # Create data loader for mini-batch training
        if hidden_maps is not None:
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train, hidden_maps_train)
        else:
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
        
        # Define map loss weight - adjust this to control importance
        map_loss_weight = 0.01
        
        # Use tqdm for progress tracking
        for epoch in tqdm(range(self.nn_epochs), desc=f"NN training (obj {objective_idx+1})"):
            # Training phase
            model.train()
            train_loss = 0.0
            
            # Handle different batch structures based on hidden map presence
            if hidden_maps is not None:
                for batch_X, batch_y, batch_maps in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass - returns both predictions
                    outputs, predicted_maps = model(batch_X)
                    # Calculate losses
                    main_loss = mse_criterion(outputs, batch_y)
                    map_loss = torch.mean((predicted_maps - batch_maps) ** 2)
                    
                    # Combined loss
                    loss = main_loss + map_loss_weight * map_loss
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)
            else:
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = mse_criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                if hidden_maps is not None:
                    # Model returns both outputs and maps
                    val_outputs, val_predicted_maps = model(X_val)
                    
                    # Calculate losses
                    val_main_loss = mse_criterion(val_outputs, y_val)
                    val_map_loss = torch.mean((val_predicted_maps - hidden_maps_val) ** 2)
                    
                    # Combined loss
                    val_loss = val_main_loss + map_loss_weight * val_map_loss
                else:
                    val_outputs = model(X_val)
                    val_loss = mse_criterion(val_outputs, y_val).item()
                
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 100 == 0 or epoch == 0:
                if hidden_maps is not None:
                    print(f"Epoch {epoch+1}/{self.nn_epochs}, Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f} (Main: {val_main_loss:.6f}, Map: {val_map_loss:.6f})")
                else:
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
        
    def _update_model(self, output_dir=None):
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
            if self.train_hidden_maps is not None:
                nn_model = self._train_neural_network(X, y.unsqueeze(-1), input_dim, i, self.train_hidden_maps)  # Pass 2D for training
            else:
                nn_model = self._train_neural_network(X, y.unsqueeze(-1), input_dim, i)  # Pass 2D for training
            self.nn_models.append(nn_model)
            
            # Create GP model with neural network mean
            model = NeuralNetworkGP(
                train_X=X,
                train_Y=y,  # Pass 1D tensor as BoTorch expects
                nn_model=nn_model,
                # outcome_transform=Standardize(m=1)
            )
            # breakpoint()
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
        
        # Store the model for prediction plots
        self.model = model_list
        
        # Create prediction vs true plots if output_dir is provided and we have enough data
        if output_dir is not None and len(self.train_x) >= 5:
            plots_dir = os.path.join(output_dir, "model_plots")
            iter_num = len(self.timing_history['model_update'])
            # Use perturbation sampling by default to test generalization around existing points
            # And use true evaluation values (not random values)
            self.plot_true_vs_predicted(plots_dir, iter_num, use_random_values=False, sample_method="random")
            print(f"Model prediction plots saved to {plots_dir}")
        
        
        return model_list
    
    def get_model_metrics(self):
        """Return neural network training metrics"""
        return self.model_metrics 

    def ask(self, output_dir=None):
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
        model_list = self._update_model(output_dir=output_dir)
        
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

    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]], hidden_maps = None):
        """Update model with evaluated points"""
        # Update budget tracking
        self.evaluated_points += len(xs)
        print(f"Tell called with {len(xs)} points. Total evaluated: {self.evaluated_points}/{self.budget}")
        
        # Convert parameter dictionaries to tensors
        x_tensors = []
        for x in xs:
            # Extract parameter values in the correct order
            x_values = []
            for name in self.parameter_space.parameters:
                param_config = self.parameter_space.parameters[name]
                value = x[name]
                
                if param_config['type'] == 'continuous':
                    x_values.append(float(value))
                elif param_config['type'] == 'integer':
                    x_values.append(float(value))
                elif param_config['type'] == 'categorical':
                    # Map category to integer
                    cat_map = self.adapter.botorch_space['categorical_maps'][name]['reverse_map']
                    x_values.append(float(cat_map[value]))
            
            x_tensors.append(torch.tensor(x_values, dtype=torch.double))
        
        # Stack tensors
        x_tensor = torch.stack(x_tensors)
        y_tensor = torch.tensor(ys, dtype=torch.double)
        
        # Add to training data
        self.train_x = torch.cat([self.train_x, x_tensor])
        self.train_y = torch.cat([self.train_y, y_tensor])
        
        # Handle hidden maps if provided
        if hidden_maps is not None:
            # Convert numpy arrays to torch tensors
            hidden_maps_tensors = []
            for h_map in hidden_maps:
                # Check if it's a numpy array and convert to tensor
                if isinstance(h_map, np.ndarray):
                    h_map_tensor = torch.tensor(h_map, dtype=torch.float32)
                    hidden_maps_tensors.append(h_map_tensor)
                elif isinstance(h_map, torch.Tensor):
                    hidden_maps_tensors.append(h_map)
                else:
                    print(f"Unsupported hidden_map type: {type(h_map)}. Skipping.")
            
            if hidden_maps_tensors:
                # Stack all tensors
                stacked_hidden_maps = torch.stack(hidden_maps_tensors)
                
                if self.train_hidden_maps is None:
                    self.train_hidden_maps = stacked_hidden_maps
                else:
                    self.train_hidden_maps = torch.cat([self.train_hidden_maps, stacked_hidden_maps], dim=0)
                
                print(f"Added hidden maps with shape: {stacked_hidden_maps.shape}")
            else:
                print("No valid hidden maps to add.")
        else:
            print("No hidden maps provided in tell() method.")


    def recommend(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Return current Pareto front"""
        return super().recommend()
        
    def get_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front"""
        return super().get_hypervolume()

    def plot_true_vs_predicted(self, output_dir: str, iter_num: int, use_random_values=False, sample_method="random"):
        """
        Plot and save true vs predicted values for each objective using random test points.
        Evaluates model generalization by testing on unseen points from the domain.
        Plots both training data (seen) and test data (unseen) with different colors.
        Also compares neural network predictions vs GP predictions with NN as mean function.
        
        Args:
            output_dir: Directory to save plots
            iter_num: Iteration number for naming the plot
            use_random_values: If False (default), use actual test problem evaluations for true values
            sample_method: Method to sample test points - options: "sobol", "random", "perturb", "acquisition"
        """
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Make sure we have a model and nn_models
        if not hasattr(self, 'model') or self.model is None:
            print("Warning: GP models not available for plotting predictions")
            return
            
        if not hasattr(self, 'nn_models') or len(self.nn_models) == 0:
            print("Warning: Neural network models not available for plotting predictions")
            return
        
        # Generate test points by sampling from the parameter space
        n_test_points = min(100, max(50, len(self.train_x) * 2))  # Generate more test points than training points
        print(f"Generating {n_test_points} test points for prediction error evaluation using {sample_method} sampling")
        
        # Option 1: Sample using acquisition optimization
        if sample_method == "acquisition" and len(self.train_x) >= 2:
            try:
                print("Using acquisition function optimization to generate test points")
                # We'll use the same process as in the ask method but with a focus on exploration
                
                # Normalize inputs
                bounds_t = self.bounds.transpose(0, 1)
                X_baseline = normalize(self.train_x, bounds_t)
                
                # Create standard uniform bounds for optimization
                standard_bounds = torch.zeros(2, X_baseline.shape[1], dtype=torch.double)
                standard_bounds[1] = 1.0
                
                # Create Monte Carlo sampler
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
                
                # Try different acquisition functions to get diverse test points
                try:
                    # Use qEI acquisition function to identify promising regions
                    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
                    acq_func = qExpectedHypervolumeImprovement(
                        model=self.model,
                        ref_point=self.ref_point.clone().detach(),
                        partitioning=FastNondominatedPartitioning(
                            ref_point=self.ref_point.clone().detach(),
                            Y=self.train_y
                        ),
                        sampler=sampler
                    )
                    
                    # Use more restarts and raw samples for better coverage
                    n_batches = min(5, n_test_points // 10)  # Generate points in batches
                    batch_size = n_test_points // n_batches  # Each batch has this many points
                    
                    # Set optimization parameters - use more restarts for better coverage
                    n_restarts = 5
                    raw_samples = 512
                    
                    # Generate batches of test points
                    all_candidates = []
                    
                    for batch in range(n_batches):
                        try:
                            # For the first batch, use standard acquisition optimization
                            if batch == 0:
                                candidates, _ = optimize_acqf(
                                    acq_function=acq_func,
                                    bounds=standard_bounds,
                                    q=batch_size,
                                    num_restarts=n_restarts,
                                    raw_samples=raw_samples,
                                    options={"batch_limit": 5, "maxiter": 100, "ftol": 1e-5, "method": "L-BFGS-B"},
                                    sequential=True,
                                )
                            else:
                                # For subsequent batches, add some noise to existing candidates for diversity
                                base_candidates = torch.cat([X_baseline, all_candidates[0] if all_candidates else torch.tensor([])])
                                noise = torch.randn_like(base_candidates[:batch_size]) * 0.1
                                noisy_candidates = torch.clamp(base_candidates[:batch_size] + noise, 0, 1)
                                
                                # Optimize from these starting points - don't use X_pending
                                # as it's not supported in the current BoTorch version
                                candidates, _ = optimize_acqf(
                                    acq_function=acq_func,
                                    bounds=standard_bounds,
                                    q=batch_size,
                                    num_restarts=n_restarts,
                                    raw_samples=raw_samples,
                                    options={"batch_limit": 5, "maxiter": 100, "ftol": 1e-5, "method": "L-BFGS-B"},
                                    sequential=True,
                                )
                            
                            all_candidates.append(candidates)
                        except Exception as e:
                            print(f"Error generating batch {batch}: {e}")
                            # Continue with next batch
                
                    # Combine all candidates
                    if all_candidates:
                        all_candidates_tensor = torch.cat(all_candidates)
                        
                        # Unnormalize candidates
                        unnormalized_candidates = unnormalize(all_candidates_tensor.detach(), bounds_t)
                        
                        # Convert to parameter dictionaries
                        test_dicts = self._tensors_to_dicts(unnormalized_candidates)
                        
                        # Limit to n_test_points
                        test_dicts = test_dicts[:n_test_points]
                        
                        print(f"Successfully generated {len(test_dicts)} test points using acquisition function")
                    else:
                        raise ValueError("Failed to generate any points with acquisition function")
                        
                except Exception as e:
                    print(f"Error using acquisition function: {e}")
                    # Fall back to random sampling
                    sample_method = "sobol"
                    
            except Exception as e:
                print(f"Error using acquisition sampling: {e}. Falling back to Sobol sampling.")
                sample_method = "sobol"  # Fall back to Sobol
        
        # Option 2: Sample from previous training points with perturbation
        if sample_method == "perturb" and len(self.train_x) > 0:
            try:
                print("Using perturbation sampling: modifying existing training points")
                # Convert training points to parameter dictionaries
                train_dicts = []
                for x_tensor in self.train_x:
                    train_dict = self.adapter.from_framework_format(x_tensor)
                    train_dicts.append(train_dict)
                
                # Generate perturbed versions of training points
                test_dicts = []
                
                # Sample with replacement if we need more test points than training points
                indices = np.random.choice(
                    len(train_dicts), 
                    size=n_test_points, 
                    replace=(n_test_points > len(train_dicts))
                )
                
                for idx in indices:
                    base_point = train_dicts[idx]
                    perturbed_point = {}
                    
                    for name, value in base_point.items():
                        config = self.parameter_space.parameters[name]
                        
                        if config['type'] == 'continuous':
                            # Add Gaussian noise (5% of range)
                            range_width = config['bounds'][1] - config['bounds'][0]
                            noise_scale = range_width * 0.05
                            new_value = value + np.random.normal(0, noise_scale)
                            # Clip to bounds
                            new_value = max(config['bounds'][0], min(config['bounds'][1], new_value))
                            perturbed_point[name] = new_value
                            
                        elif config['type'] == 'integer':
                            # Add small integer noise
                            range_width = config['bounds'][1] - config['bounds'][0]
                            max_step = max(1, int(range_width * 0.1))  # 10% of range, at least 1
                            step = np.random.randint(-max_step, max_step+1)
                            new_value = int(value) + step
                            # Clip to bounds
                            new_value = max(config['bounds'][0], min(config['bounds'][1], new_value))
                            perturbed_point[name] = new_value
                            
                        elif config['type'] == 'categorical':
                            # 20% chance to change category
                            if np.random.random() < 0.2:
                                if 'categories' in config:
                                    categories = config['categories']
                                else:
                                    categories = config['values']
                                # Choose a different category with equal probability
                                other_categories = [c for c in categories if c != value]
                                if other_categories:
                                    perturbed_point[name] = np.random.choice(other_categories)
                                else:
                                    perturbed_point[name] = value
                            else:
                                perturbed_point[name] = value
                    
                    test_dicts.append(perturbed_point)
                    
                print(f"Successfully generated {len(test_dicts)} perturbed test points")
                
            except Exception as e:
                print(f"Error using perturbation sampling: {e}. Falling back to Sobol sampling.")
                sample_method = "sobol"  # Fall back to Sobol
        
        # Option 3: Use pure random sampling (fallback)
        if sample_method == "random":
            print("Using uniform random sampling")
            test_dicts = []
            for _ in range(n_test_points):
                test_point = {}
                # Sample each parameter type appropriately
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        # Sample continuous uniform
                        test_point[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        # Sample integer uniform
                        test_point[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        # Sample categorical uniform
                        if 'categories' in config:
                            test_point[name] = np.random.choice(config['categories'])
                        else:
                            test_point[name] = np.random.choice(config['values'])
                test_dicts.append(test_point)
        
        # Convert parameter dictionaries to tensors for model prediction
        test_tensors = []
        for x in test_dicts:
            # Extract parameter values in the correct order
            x_values = []
            for name in self.parameter_space.parameters:
                param_config = self.parameter_space.parameters[name]
                value = x[name]
                
                if param_config['type'] == 'continuous':
                    x_values.append(float(value))
                elif param_config['type'] == 'integer':
                    x_values.append(float(value))
                elif param_config['type'] == 'categorical':
                    # Map category to integer
                    cat_map = self.adapter.botorch_space['categorical_maps'][name]['reverse_map']
                    x_values.append(float(cat_map[value]))
            
            test_tensors.append(torch.tensor(x_values, dtype=torch.double))
        
        # Stack tensors
        X_test_tensor = torch.stack(test_tensors)
        


        #####
        known_problem = None
        known_problem = 'complex_categorical'
        
        # Get true values by evaluating the test problem or generating random data
        if use_random_values:
            print("Using random values for true evaluations")
            true_np = np.random.rand(n_test_points, self.n_objectives) * 10
        else:
            # Try to determine and use the actual test problem
            try:
                from tests.test_algorithms.test_problems import get_test_problem, TEST_PROBLEMS
                
                # Find the most likely test problem by checking parameter space similarity
                problem_name = "unknown"
                best_match = 0
                best_param_match = {}
                
                # Loop through all test problems to find the best match
                for name, problem in TEST_PROBLEMS.items():
                    problem_space = problem.get_parameter_space()
                    our_param_names = set(self.parameter_space.parameters.keys())
                    problem_param_names = set(problem_space.parameters.keys())
                    
                    if known_problem:
                        if name == known_problem:
                            problem_name = name
                            best_param_match = {
                                'our_params': our_param_names,
                                'problem_params': problem_param_names
                            }
                            break

                    # Check if our parameters are a subset of the test problem's parameters
                    if our_param_names.issubset(problem_param_names):
                        matches = len(our_param_names)
                        if matches > best_match:
                            best_match = matches
                            problem_name = name
                            best_param_match = {
                                'our_params': our_param_names,
                                'problem_params': problem_param_names
                            }
                if problem_name == "unknown":
                    raise ValueError("No matching test problem found for parameter space")
                    
                print(f"Using test problem '{problem_name}' for evaluating prediction accuracy")
                print(f"Our parameters: {len(best_param_match['our_params'])}, Test problem parameters: {len(best_param_match['problem_params'])}")

                test_problem = get_test_problem(problem_name)
                
                # Evaluate test points using the test problem
                true_values = []
                try:
                    # Get all parameters required by the test problem
                    problem_params = set(problem_space.parameters.keys())
                    our_params = set(self.parameter_space.parameters.keys())
                    
                    # Handle missing parameters
                    missing_params = problem_params - our_params
                    if missing_params:
                        print(f"Missing parameters that will be filled with defaults: {missing_params}")
                    
                    for test_dict in test_dicts:
                        # Create a complete parameter dict by adding default values for missing parameters
                        complete_test_dict = test_dict.copy()
                        for param in missing_params:
                            param_config = problem_space.parameters[param]
                            if param_config['type'] == 'continuous':
                                # Use the middle of the range as default
                                complete_test_dict[param] = (param_config['bounds'][0] + param_config['bounds'][1]) / 2
                            elif param_config['type'] == 'integer':
                                # Use the middle of the range as default
                                complete_test_dict[param] = int((param_config['bounds'][0] + param_config['bounds'][1]) // 2)
                            elif param_config['type'] == 'categorical':
                                # Use the first category as default
                                if 'categories' in param_config:
                                    complete_test_dict[param] = param_config['categories'][0]
                                else:
                                    complete_test_dict[param] = param_config['values'][0]
                                    
                        # Evaluate with the complete parameter set
                        test_result = test_problem.evaluate(complete_test_dict)
                        if type(test_result) == tuple:
                            true_values.append(test_result[0])
                        else:
                            true_values.append(test_result)
                    
                    true_np = np.array(true_values)
                except Exception as e:
                    print(f"Error evaluating test points: {e}")
                    breakpoint()
                    print("Falling back to random values for true evaluations")
                    true_np = np.random.rand(n_test_points, self.n_objectives) * 10
                    
            except Exception as e:
                print(f"Could not use test problem for evaluation: {e}")
                print("Falling back to random values for true evaluations")
                true_np = np.random.rand(n_test_points, self.n_objectives) * 10
        
        # Normalize inputs for both NN and GP prediction
        X_norm_test = normalize(X_test_tensor, self.bounds.transpose(0, 1))
        
        # Get GP predictions for the test points
        with torch.no_grad():
            gp_predictions_test = self.model.posterior(X_norm_test).mean
            
        # Get NN-only predictions for test points
        nn_predictions_test = []
        for i, nn_model in enumerate(self.nn_models):
            with torch.no_grad():
                # Use the neural network directly for predictions
                nn_pred = nn_model(X_norm_test)
                if type(nn_pred) == tuple:
                    nn_pred, hidden_map = nn_pred

                if type(nn_pred) == tuple:
                    nn_pred, hidden_map = nn_pred
                # Make sure output is the right shape
                if nn_pred.dim() > 1 and nn_pred.shape[1] == 1:
                    nn_pred = nn_pred.squeeze(-1)
                nn_predictions_test.append(nn_pred)
        
        # Stack neural network predictions along a new dimension
        nn_pred_tensor = torch.stack(nn_predictions_test, dim=1)
        
        # Convert predictions to numpy for easier handling
        gp_pred_test_np = gp_predictions_test.cpu().numpy()
        nn_pred_test_np = nn_pred_tensor.cpu().numpy()
        
        # Get predictions for the training points
        with torch.no_grad():
            X_norm_train = normalize(self.train_x, self.bounds.transpose(0, 1))
            gp_predictions_train = self.model.posterior(X_norm_train).mean
            
            # Get NN-only predictions for training points
            nn_predictions_train = []
            for i, nn_model in enumerate(self.nn_models):
                # Use the neural network directly for predictions
                nn_pred = nn_model(X_norm_train)
                if type(nn_pred) == tuple:
                    nn_pred, hidden_map = nn_pred
                # Make sure output is the right shape
                if nn_pred.dim() > 1 and nn_pred.shape[1] == 1:
                    nn_pred = nn_pred.squeeze(-1)
                nn_predictions_train.append(nn_pred)
                
            # Stack neural network predictions along a new dimension
            nn_pred_train_tensor = torch.stack(nn_predictions_train, dim=1)
        
        # Convert predictions and true values to numpy
        gp_pred_train_np = gp_predictions_train.cpu().numpy()
        nn_pred_train_np = nn_pred_train_tensor.cpu().numpy()
        true_train_np = self.train_y.cpu().numpy()
        
        # Plot for each objective
        for i in range(self.n_objectives):
            # Create figure with larger size for combined plot
            plt.figure(figsize=(16, 12))
            
            # Subplot for GP predictions
            plt.subplot(2, 2, 1)
            plt.title(f'GP Predictions - Training Data (Objective {i+1})')
            plt.scatter(true_train_np[:, i], gp_pred_train_np[:, i], alpha=0.7, color='blue', 
                       marker='o')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_train_np[:, i]), np.min(gp_pred_train_np[:, i]))
            max_val = max(np.max(true_train_np[:, i]), np.max(gp_pred_train_np[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.xlabel('True values')
            plt.ylabel('GP predicted values')
            
            # Calculate error metrics
            gp_train_mse = np.mean((true_train_np[:, i] - gp_pred_train_np[:, i])**2)
            gp_train_mae = np.mean(np.abs(true_train_np[:, i] - gp_pred_train_np[:, i]))
            plt.text(0.05, 0.95, f'MSE: {gp_train_mse:.4f}\nMAE: {gp_train_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Subplot for NN predictions
            plt.subplot(2, 2, 2)
            plt.title(f'NN Predictions - Training Data (Objective {i+1})')
            plt.scatter(true_train_np[:, i], nn_pred_train_np[:, i], alpha=0.7, color='green', 
                       marker='o')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_train_np[:, i]), np.min(nn_pred_train_np[:, i]))
            max_val = max(np.max(true_train_np[:, i]), np.max(nn_pred_train_np[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.xlabel('True values')
            plt.ylabel('NN predicted values')
            
            # Calculate error metrics
            nn_train_mse = np.mean((true_train_np[:, i] - nn_pred_train_np[:, i])**2)
            nn_train_mae = np.mean(np.abs(true_train_np[:, i] - nn_pred_train_np[:, i]))
            plt.text(0.05, 0.95, f'MSE: {nn_train_mse:.4f}\nMAE: {nn_train_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Subplot for GP test predictions
            plt.subplot(2, 2, 3)
            plt.title(f'GP Predictions - Test Data (Objective {i+1})')
            plt.scatter(true_np[:, i], gp_pred_test_np[:, i], alpha=0.7, color='red', 
                       marker='x')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_np[:, i]), np.min(gp_pred_test_np[:, i]))
            max_val = max(np.max(true_np[:, i]), np.max(gp_pred_test_np[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.xlabel('True values')
            plt.ylabel('GP predicted values')
            
            # Calculate error metrics
            gp_test_mse = np.mean((true_np[:, i] - gp_pred_test_np[:, i])**2)
            gp_test_mae = np.mean(np.abs(true_np[:, i] - gp_pred_test_np[:, i]))
            plt.text(0.05, 0.95, f'MSE: {gp_test_mse:.4f}\nMAE: {gp_test_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Subplot for NN test predictions
            plt.subplot(2, 2, 4)
            plt.title(f'NN Predictions - Test Data (Objective {i+1})')
            plt.scatter(true_np[:, i], nn_pred_test_np[:, i], alpha=0.7, color='orange', 
                       marker='x')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_np[:, i]), np.min(nn_pred_test_np[:, i]))
            max_val = max(np.max(true_np[:, i]), np.max(nn_pred_test_np[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.xlabel('True values')
            plt.ylabel('NN predicted values')
            
            # Calculate error metrics
            nn_test_mse = np.mean((true_np[:, i] - nn_pred_test_np[:, i])**2)
            nn_test_mae = np.mean(np.abs(true_np[:, i] - nn_pred_test_np[:, i]))
            plt.text(0.05, 0.95, f'MSE: {nn_test_mse:.4f}\nMAE: {nn_test_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Add overall title
            plt.suptitle(f'GP vs. NN Predictions - Objective {i+1}\n' +
                        f'Training: GP MSE={gp_train_mse:.4f}, NN MSE={nn_train_mse:.4f}\n' +
                        f'Test: GP MSE={gp_test_mse:.4f}, NN MSE={nn_test_mse:.4f}', 
                        fontsize=14)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Make room for suptitle
            
            # Save the combined plot
            plt.savefig(os.path.join(output_dir, f'gp_vs_nn_obj_{i+1}_iter_{iter_num}.png'), dpi=150)
            plt.close()
            
            # Also create a combined plot - both models on same axes for direct comparison
            plt.figure(figsize=(15, 10))
            
            # Training data
            plt.subplot(1, 2, 1)
            plt.title(f'Training Data - Objective {i+1}')
            plt.scatter(true_train_np[:, i], gp_pred_train_np[:, i], alpha=0.7, color='blue', 
                       marker='o', label='GP predictions')
            plt.scatter(true_train_np[:, i], nn_pred_train_np[:, i], alpha=0.7, color='green', 
                       marker='^', label='NN predictions')
            
            # Find global min and max for diagonal line
            global_min = min(np.min(true_train_np[:, i]), np.min(gp_pred_train_np[:, i]), np.min(nn_pred_train_np[:, i]))
            global_max = max(np.max(true_train_np[:, i]), np.max(gp_pred_train_np[:, i]), np.max(nn_pred_train_np[:, i]))
            plt.plot([global_min, global_max], [global_min, global_max], 'k--', label='Perfect prediction')
            
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
            
            # Error metrics text
            plt.text(0.05, 0.95, 
                    f'GP - MSE: {gp_train_mse:.4f}, MAE: {gp_train_mae:.4f}\n' +
                    f'NN - MSE: {nn_train_mse:.4f}, MAE: {nn_train_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Test data
            plt.subplot(1, 2, 2)
            plt.title(f'Test Data - Objective {i+1}')
            plt.scatter(true_np[:, i], gp_pred_test_np[:, i], alpha=0.7, color='red', 
                       marker='x', label='GP predictions')
            plt.scatter(true_np[:, i], nn_pred_test_np[:, i], alpha=0.7, color='orange', 
                       marker='+', label='NN predictions')
            
            # Find global min and max for diagonal line
            global_min = min(np.min(true_np[:, i]), np.min(gp_pred_test_np[:, i]), np.min(nn_pred_test_np[:, i]))
            global_max = max(np.max(true_np[:, i]), np.max(gp_pred_test_np[:, i]), np.max(nn_pred_test_np[:, i]))
            plt.plot([global_min, global_max], [global_min, global_max], 'k--', label='Perfect prediction')
            
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
            
            # Error metrics text
            plt.text(0.05, 0.95, 
                    f'GP - MSE: {gp_test_mse:.4f}, MAE: {gp_test_mae:.4f}\n' +
                    f'NN - MSE: {nn_test_mse:.4f}, MAE: {nn_test_mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Add overall title
            plt.suptitle(f'GP vs. NN Predictions Comparison - Objective {i+1}', fontsize=14)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for suptitle
            
            # Save the overlay comparison plot
            plt.savefig(os.path.join(output_dir, f'comparison_obj_{i+1}_iter_{iter_num}.png'), dpi=150)
            plt.close()

    def plot_and_save_model_error(self, output_dir: str):
        """Plot and save model error (train/val loss) per epoch to output_dir."""
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        metrics = self.get_model_metrics()
        train_losses = metrics.get('train_losses', [])
        val_losses = metrics.get('val_losses', [])
        for i, (train, val) in enumerate(zip(train_losses, val_losses)):
            plt.figure()
            plt.plot(train, label='Train Loss')
            plt.plot(val, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Model Error (Objective {i+1})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'model_error_objective_{i+1}.png'))
            plt.close() 