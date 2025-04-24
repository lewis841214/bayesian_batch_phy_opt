import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd

import botorch
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch.nn import Sequential, Linear, ReLU, Dropout, MSELoss
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from pymoo.indicators.hv import HV
import time
from tqdm import tqdm

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.botorch_adapter import BoTorchAdapter

# Custom Neural Network surrogate model compatible with BoTorch
class NNSurrogateModel(Model):
    def __init__(self, input_dim, output_dim=1, hidden_dim=50, dropout_rate=0.1, ensemble_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.ensemble_size = ensemble_size
        
        # Create ensemble of networks
        self.networks = [
            Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                Dropout(dropout_rate),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Dropout(dropout_rate),
                Linear(hidden_dim, output_dim),
            ) for _ in range(ensemble_size)
        ]
        
        # Move networks to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for net in self.networks:
            net.to(self.device)
            
        # Optimization setup
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=0.01) for net in self.networks]
        self.loss_fn = MSELoss()
        
        # Data normalization
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Training data
        self.train_x = None
        self.train_y = None
        
    def posterior(self, X, observation_noise=False, posterior_transform=None):
        """Return posterior distribution at X"""
        # Ensure X is on the correct device
        X = X.to(self.device)
        
        # Normalize inputs
        X_np = X.cpu().numpy()
        X_norm = torch.tensor(self.x_scaler.transform(X_np), dtype=torch.float32).to(self.device)
        
        # Get predictions from all ensemble members
        with torch.no_grad():
            preds = torch.stack([net(X_norm) for net in self.networks])
        
        # Calculate mean and variance across ensemble
        mean = preds.mean(dim=0)
        variance = preds.var(dim=0) + 1e-6  # Add small constant for numerical stability
        
        # Rescale outputs
        mean_np = mean.cpu().numpy()
        mean_rescaled = torch.tensor(self.y_scaler.inverse_transform(mean_np), dtype=X.dtype).to(X.device)
        variance_rescaled = torch.tensor(
            variance.cpu().numpy() * (self.y_scaler.scale_**2),
            dtype=X.dtype
        ).to(X.device)
        
        # Create a GPyTorch-compatible posterior
        covar_matrix = torch.diag_embed(variance_rescaled.squeeze(-1))
        mvn = MultivariateNormal(mean_rescaled, covar_matrix)
        posterior = GPyTorchPosterior(mvn)
        
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
    
    def fit(self, train_x, train_y, epochs=100, batch_size=32, verbose=False):
        """Fit the model to training data"""
        # Save the training data
        self.train_x = train_x.cpu().numpy()
        self.train_y = train_y.cpu().numpy()
        
        # Normalize the data
        self.x_scaler.fit(self.train_x)
        self.y_scaler.fit(self.train_y)
        
        X_norm = torch.tensor(self.x_scaler.transform(self.train_x), dtype=torch.float32).to(self.device)
        y_norm = torch.tensor(self.y_scaler.transform(self.train_y), dtype=torch.float32).to(self.device)
        
        # Train each network in the ensemble
        for i, (net, optimizer) in enumerate(zip(self.networks, self.optimizers)):
            net.train()
            
            # Training loop
            for epoch in range(epochs):
                # Generate random indices for mini-batches
                idx = torch.randperm(X_norm.shape[0])
                
                # Mini-batch training
                batch_losses = []
                for j in range(0, X_norm.shape[0], batch_size):
                    batch_idx = idx[j:min(j+batch_size, X_norm.shape[0])]
                    X_batch = X_norm[batch_idx]
                    y_batch = y_norm[batch_idx]
                    
                    optimizer.zero_grad()
                    pred = net(X_batch)
                    loss = self.loss_fn(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                if verbose and (epoch+1) % 20 == 0:
                    print(f"Network {i+1}/{self.ensemble_size}, Epoch {epoch+1}/{epochs}, Loss: {np.mean(batch_losses):.4f}")
        
        return self
    
    def condition_on_observations(self, X, Y, **kwargs):
        """Update the model with new observations (not used by BoTorch)"""
        self.train_x = np.vstack([self.train_x, X.cpu().numpy()])
        self.train_y = np.vstack([self.train_y, Y.cpu().numpy()])
        self.fit(torch.tensor(self.train_x), torch.tensor(self.train_y))
        return self
    
    def subset_output(self, idcs):
        """Get a model with a subset of outputs"""
        raise NotImplementedError("Subset output not supported for NN surrogate model")

# Custom XGBoost surrogate model compatible with BoTorch
class XGBoostSurrogateModel(Model):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        # Create base model with uncertainty via quantile regression
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        # For uncertainty estimation, we use a second model to predict squared errors
        self.uncertainty_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        # Data normalization
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Training data
        self.train_x = None
        self.train_y = None
        
    def posterior(self, X, observation_noise=False, posterior_transform=None):
        """Return posterior distribution at X"""
        # Convert to numpy for XGBoost
        X_np = X.cpu().numpy()
        X_norm = self.x_scaler.transform(X_np)
        
        # Make predictions with the model
        mean_norm = self.model.predict(X_norm)
        
        # Predict uncertainty (variance)
        variance_norm = np.maximum(1e-6, self.uncertainty_model.predict(X_norm))
        
        # Rescale outputs
        mean = self.y_scaler.inverse_transform(mean_norm.reshape(-1, 1)).flatten()
        variance = variance_norm * (self.y_scaler.scale_**2)
        
        # Convert to torch tensors
        mean_t = torch.tensor(mean.reshape(-1, 1), dtype=X.dtype, device=X.device)
        variance_t = torch.tensor(variance.reshape(-1, 1), dtype=X.dtype, device=X.device)
        
        # Create a GPyTorch-compatible posterior
        covar_matrix = torch.diag_embed(variance_t.squeeze(-1))
        mvn = MultivariateNormal(mean_t, covar_matrix)
        posterior = GPyTorchPosterior(mvn)
        
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
    
    def fit(self, train_x, train_y):
        """Fit the model to training data"""
        # Save the training data
        self.train_x = train_x.cpu().numpy()
        self.train_y = train_y.cpu().numpy()
        
        # Normalize the data
        self.x_scaler.fit(self.train_x)
        self.y_scaler.fit(self.train_y)
        
        X_norm = self.x_scaler.transform(self.train_x)
        y_norm = self.y_scaler.transform(self.train_y).flatten()
        
        # Fit the mean model
        self.model.fit(X_norm, y_norm)
        
        # Make predictions to compute squared errors for the uncertainty model
        y_pred = self.model.predict(X_norm)
        squared_errors = (y_norm - y_pred) ** 2
        
        # Fit the uncertainty model on squared errors
        self.uncertainty_model.fit(X_norm, squared_errors)
        
        return self
    
    def condition_on_observations(self, X, Y, **kwargs):
        """Update the model with new observations (not used in this implementation)"""
        self.train_x = np.vstack([self.train_x, X.cpu().numpy()])
        self.train_y = np.vstack([self.train_y, Y.cpu().numpy()])
        self.fit(torch.tensor(self.train_x), torch.tensor(self.train_y))
        return self
    
    def subset_output(self, idcs):
        """Get a model with a subset of outputs"""
        raise NotImplementedError("Subset output not supported for XGBoost surrogate model")

class QNEHVIHybrid(MultiObjectiveOptimizer):
    """
    q-Noisy Expected Hypervolume Improvement (qNEHVI) implementation using BoTorch
    with neural network or XGBoost surrogate models instead of GPs
    
    This hybrid approach allows for potentially better modeling of complex landscapes
    or large-scale problems where GPs might be computationally expensive.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        budget: int,
        batch_size: int = 1, 
        n_objectives: int = 2,
        surrogate_model: str = "nn",  # Options: "nn" or "xgboost"
        ref_point: Optional[List[float]] = None,
        noise_std: Optional[List[float]] = None,
        mc_samples: int = 128,
        **kwargs
    ):
        """
        Initialize the qNEHVI optimizer with hybrid models
        
        Args:
            parameter_space: Parameter space to optimize
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            n_objectives: Number of objectives to optimize
            surrogate_model: Type of surrogate model to use ("nn" or "xgboost")
            ref_point: Reference point for hypervolume calculation
            noise_std: Standard deviation of observation noise for each objective
            mc_samples: Number of MC samples for acquisition function approximation
        """
        self.mc_samples = mc_samples
        self.ref_point = ref_point
        self.noise_std = noise_std
        self.surrogate_model_type = surrogate_model.lower()
        
        # Default NN parameters
        self.nn_hidden_dim = kwargs.get("nn_hidden_dim", 50)
        self.nn_dropout_rate = kwargs.get("nn_dropout_rate", 0.1)
        self.nn_ensemble_size = kwargs.get("nn_ensemble_size", 5)
        self.nn_epochs = kwargs.get("nn_epochs", 100)
        
        # Default XGBoost parameters
        self.xgb_n_estimators = kwargs.get("xgb_n_estimators", 100)
        self.xgb_max_depth = kwargs.get("xgb_max_depth", 3)
        self.xgb_learning_rate = kwargs.get("xgb_learning_rate", 0.1)
        
        super().__init__(parameter_space, budget, batch_size, n_objectives)
        
        # Track timings for progress estimation
        self.timing_history = {
            'model_update': [],
            'acquisition_optimization': [],
            'iteration': []
        }
        
        # Track evaluations and remaining budget
        self.evaluated_points = 0
        
        # Set default reference point if not provided
        if self.ref_point is None:
            self.ref_point = torch.ones(self.n_objectives) * 10.0
        else:
            # Convert to tensor and ensure proper shape
            if isinstance(self.ref_point, list):
                self.ref_point = torch.tensor(self.ref_point, dtype=torch.double).view(-1)
            else:
                self.ref_point = self.ref_point.view(-1)
    
    def _setup(self):
        """Initialize BoTorch components"""
        self.adapter = BoTorchAdapter(self.parameter_space)
        self.bounds = self.adapter.botorch_space['bounds']
        self.train_x = torch.empty((0, len(self.parameter_space.parameters)), dtype=torch.double)
        self.train_y = torch.empty((0, self.n_objectives), dtype=torch.double)
        
        # Define parameter space dimension
        self.dim = len(self.parameter_space.parameters)
        
        # Initialize model
        self.models = None
        
    def _update_model(self):
        """Update the surrogate model with current training data"""
        start_time = time.time()
        
        if len(self.train_x) < 2:
            # Not enough data to fit a model
            return None
            
        # Normalize inputs - fix bounds format
        bounds_t = self.bounds.transpose(0, 1)  # Make it (n_dims, 2)
        X = normalize(self.train_x, bounds_t)
        Y = self.train_y
        
        print(f"Fitting {self.surrogate_model_type.upper()} model with {len(X)} observations...")
        
        # Create and fit a model for each objective
        models = []
        
        for i in tqdm(range(self.n_objectives), desc=f"Fitting {self.surrogate_model_type.upper()} models"):
            y = Y[:, i:i+1]  # Get ith objective, keep dimension
            
            if self.surrogate_model_type == "nn":
                model = NNSurrogateModel(
                    input_dim=X.shape[1],
                    hidden_dim=self.nn_hidden_dim,
                    dropout_rate=self.nn_dropout_rate,
                    ensemble_size=self.nn_ensemble_size
                )
                model.fit(X, y, epochs=self.nn_epochs, verbose=(i==0))  # Only show verbose output for first objective
            
            elif self.surrogate_model_type == "xgboost":
                model = XGBoostSurrogateModel(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    learning_rate=self.xgb_learning_rate
                )
                model.fit(X, y)
            
            else:
                raise ValueError(f"Unknown surrogate model type: {self.surrogate_model_type}")
                
            models.append(model)
            
        # Create a ModelListGP-like structure (not using ModelListGP since our models aren't GPs)
        model_list = models
            
        elapsed = time.time() - start_time
        self.timing_history['model_update'].append(elapsed)
        print(f"Model fitting completed in {elapsed:.2f} seconds")
        
        return model_list
    
    def ask(self):
        """Return a batch of points to evaluate"""
        # Calculate remaining budget and effective batch size
        remaining = self.budget - self.evaluated_points
        effective_batch_size = min(self.batch_size, remaining)
        
        if effective_batch_size <= 0:
            print("Budget exhausted, no more evaluations possible.")
            return []
            
        print(f"Generating batch of {effective_batch_size} candidates. Remaining budget: {remaining}")
        
        # For debugging
        print("Parameter space:")
        for name, config in self.parameter_space.parameters.items():
            print(f"  {name}: {config['type']}")
            if config['type'] == 'categorical':
                print(f"     values: {config.get('categories', [])}")
        
        # If we don't have enough points, generate random samples
        if len(self.train_x) < 2 * self.n_objectives:
            print("Insufficient data for surrogate model. Using random sampling.")
            
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
                        # Get the available categories
                        categories = config.get('categories', [])
                        if not categories:
                            categories = config.get('values', [])
                        
                        # Ensure we have categories to sample from
                        if not categories:
                            raise ValueError(f"No categories defined for parameter {name}")
                            
                        # Convert to regular Python string to avoid numpy string type issues
                        random_params[name] = str(np.random.choice(categories))
                candidates.append(random_params)
                
            return candidates
        
        # Update surrogate models
        model_list = self._update_model()
        
        if model_list is None:
            # Fall back to random sampling
            # Generate candidates directly using the parameter space
            candidates = []
            for _ in range(effective_batch_size):
                random_params = {}
                for name, config in self.parameter_space.parameters.items():
                    if config['type'] == 'continuous':
                        random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                    elif config['type'] == 'integer':
                        random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    elif config['type'] == 'categorical':
                        # Get the available categories
                        categories = config.get('categories', [])
                        if not categories:
                            categories = config.get('values', [])
                        
                        # Ensure we have categories to sample from
                        if not categories:
                            raise ValueError(f"No categories defined for parameter {name}")
                            
                        # Convert to regular Python string to avoid numpy string type issues
                        random_params[name] = str(np.random.choice(categories))
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
            
            # Add detailed debugging information about tensor shapes
            print(f"DEBUG - Train_y shape: {self.train_y.shape}")
            print(f"DEBUG - Ref_point before adjustment: {self.ref_point}, shape: {self.ref_point.shape}")
            
            # Adjust reference point to match actual number of objectives if needed
            y_dim = self.train_y.shape[1]
            if self.ref_point.shape[0] != y_dim:
                print(f"Warning: Reference point dimensions ({self.ref_point.shape[0]}) don't match problem objectives ({y_dim}). Adjusting.")
                if y_dim > self.ref_point.shape[0]:
                    # Expand reference point with default values
                    new_ref = torch.ones(y_dim, dtype=torch.double) * 11.0
                    new_ref[:self.ref_point.shape[0]] = self.ref_point
                    self.ref_point = new_ref
                else:
                    # Truncate reference point
                    self.ref_point = self.ref_point[:y_dim]
            
            print(f"DEBUG - Ref_point after adjustment: {self.ref_point}, shape: {self.ref_point.shape}")
            print(f"DEBUG - X_baseline shape: {X_baseline.shape}")
            
            # Debug the mean and variance from the model wrapper
            print("DEBUG - Testing model output shapes:")
            
            # Create a wrapper for our models that looks like ModelListGP to BoTorch
            class SingleModelWrapper(Model):
                """Wrapper for a single model to ensure proper posterior formatting"""
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    
                def posterior(self, X, observation_noise=False, **kwargs):
                    """Get posterior from the model and ensure proper formatting"""
                    try:
                        posterior = self.model.posterior(X)
                        mean = posterior.mean
                        variance = posterior.variance
                        
                        batch_size = X.shape[0]
                        
                        # KEY FIX: BoTorch expects [batch_size, output_dim] tensors, not [batch_size, 1, 1]
                        # Ensure proper shapes directly without reshaping the original tensors
                        if mean.dim() == 3 and mean.shape[1] == 1 and mean.shape[2] == 1:
                            # Shape is [batch_size, 1, 1] - flatten to [batch_size, 1]
                            mean = mean.squeeze(-1)
                        elif mean.dim() == 1:
                            # Shape is [batch_size] - expand to [batch_size, 1]
                            mean = mean.unsqueeze(-1)
                            
                        if variance.dim() == 3 and variance.shape[1] == 1 and variance.shape[2] == 1:
                            # Shape is [batch_size, 1, 1] - flatten to [batch_size, 1]
                            variance = variance.squeeze(-1)
                        elif variance.dim() == 1:
                            # Shape is [batch_size] - expand to [batch_size, 1]
                            variance = variance.unsqueeze(-1)
                        
                        # Print the shapes for debugging
                        print(f"DEBUG - Formatted mean shape: {mean.shape}, variance shape: {variance.shape}")
                        
                        # Use properly formatted tensors to create MVN
                        # For BoTorch, we need a batch MVN with event shape [1]
                        # Create a proper batched covariance matrix with shape [batch_size, 1, 1]
                        covar_matrix = torch.zeros((batch_size, 1, 1), dtype=X.dtype, device=X.device)
                        for i in range(batch_size):
                            covar_matrix[i, 0, 0] = variance[i, 0]
                        
                        # Create the MVN
                        mvn = MultivariateNormal(mean, covar_matrix)
                        
                        # Return a GPyTorchPosterior to ensure compatibility with BoTorch
                        return GPyTorchPosterior(mvn)
                        
                    except Exception as e:
                        print(f"Error in SingleModelWrapper: {e}")
                        # Create a simple fallback posterior with the correct shape
                        batch_size = X.shape[0]
                        simple_mean = torch.zeros((batch_size, 1), dtype=X.dtype, device=X.device)
                        simple_covar = torch.eye(1, dtype=X.dtype, device=X.device).expand(batch_size, 1, 1)
                        simple_mvn = MultivariateNormal(simple_mean, simple_covar)
                        return GPyTorchPosterior(simple_mvn)
            
            class ModelListWrapper(Model):
                def __init__(self, models):
                    super().__init__()
                    # Wrap each model individually to ensure proper posterior formatting
                    self.models = [SingleModelWrapper(model) for model in models]
                    
                def posterior(self, X, observation_noise=False, **kwargs):
                    """Return a combined posterior"""
                    batch_size = X.shape[0]
                    num_models = len(self.models)
                    print(f"DEBUG - Processing posterior for batch size: {batch_size}, models: {num_models}")
                    
                    # Get posteriors from all models
                    try:
                        posteriors = [model.posterior(X) for model in self.models]
                    except Exception as e:
                        print(f"ERROR in model posterior: {e}")
                        print(f"Input tensor shape: {X.shape}")
                        raise ValueError(f"Failed to get posterior from models: {e}")
                    
                    # Debug posterior shapes
                    print(f"DEBUG - Number of posteriors: {len(posteriors)}")
                    for i, p in enumerate(posteriors):
                        print(f"DEBUG - Posterior {i} mean shape: {p.mean.shape}, variance shape: {p.variance.shape}")
                    
                    # Extract and correctly format means and variances
                    try:
                        # KEY FIX: Create a joint posterior that BoTorch expects for multi-objective optimization
                        # We need mean shape [batch_size, num_objectives] and proper covariance
                        
                        # Set up correctly shaped output tensors
                        joint_mean = torch.zeros(batch_size, num_models, dtype=X.dtype, device=X.device)
                        
                        # Carefully extract means from each posterior and ensure they have correct shape
                        for i, p in enumerate(posteriors):
                            m = p.mean
                            # Convert to consistent shape [batch_size, 1]
                            if m.dim() == 3:  # [batch, 1, 1]
                                m = m.squeeze(-1).squeeze(-1)
                            elif m.dim() == 2 and m.shape[1] == 1:  # [batch, 1]
                                m = m.squeeze(-1)
                                
                            # Copy to output tensor
                            joint_mean[:, i] = m
                            
                        # Create independent batched joint covariance (no correlation between objectives)
                        # Shape needed: [batch_size, num_objectives, num_objectives]
                        joint_covar = torch.zeros(batch_size, num_models, num_models, 
                                            dtype=X.dtype, device=X.device)
                        
                        # Fill in diagonal entries with variances
                        for b in range(batch_size):
                            for i in range(num_models):
                                var = posteriors[i].variance
                                
                                # Extract variance properly based on its shape
                                if var.dim() == 3:  # [batch, 1, 1]
                                    v = var[b, 0, 0]
                                elif var.dim() == 2:  # [batch, 1]
                                    v = var[b, 0]
                                else:  # [batch]
                                    v = var[b]
                                    
                                # Set diagonal element
                                joint_covar[b, i, i] = v
                        
                        # Create the multivariate normal distribution
                        joint_mvn = MultivariateNormal(joint_mean, joint_covar)
                        
                        # Create the posterior
                        joint_posterior = GPyTorchPosterior(joint_mvn)
                        
                        # Debug the joint posterior
                        print(f"DEBUG - Joint posterior mean shape: {joint_posterior.mean.shape}")
                        print(f"DEBUG - Joint posterior variance shape: {joint_posterior.variance.shape}")
                        
                        return joint_posterior
                        
                    except Exception as e:
                        print(f"ERROR combining posteriors: {e}")
                        print(f"Full error details: {str(e)}")
                        
                        # Fallback: Create a simple posterior with the right shape
                        dummy_mean = torch.zeros(batch_size, num_models, dtype=X.dtype, device=X.device)
                        dummy_covar = torch.eye(num_models, dtype=X.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
                        dummy_mvn = MultivariateNormal(dummy_mean, dummy_covar)
                        return GPyTorchPosterior(dummy_mvn)
            
            model_wrapper = ModelListWrapper(model_list)
            
            # Debug the model wrapper with a test input
            print("DEBUG - Testing model_wrapper with single input:")
            with torch.no_grad():
                # Use a test input with the correct dimensionality (same as X_baseline)
                test_x = X_baseline[:1].clone()  # Take the first training example as test - this ensures correct dimensions
                try:
                    test_posterior = model_wrapper.posterior(test_x)
                    print(f"DEBUG - Test posterior mean shape: {test_posterior.mean.shape}")
                    print(f"DEBUG - Test posterior variance shape: {test_posterior.variance.shape}")
                except Exception as e:
                    print(f"Error testing model posterior: {e}")
                    print("Model test failed, but continuing with acquisition function creation")
            
            # Convert ref_point to column vector if needed
            ref_point_tensor = self.ref_point.clone().detach()
            print(f"DEBUG - Final ref_point used in acq function: {ref_point_tensor}, shape: {ref_point_tensor.shape}")
            
            # Format alpha correctly
            alpha_value = alpha if isinstance(alpha, list) else [alpha] * self.n_objectives
            print(f"DEBUG - Alpha value: {alpha_value}")
            
            # Store the model list in self.models for potential fallback use
            self.models = model_list
            
            # Create the acquisition function with the sampler
            print("DEBUG - Creating acquisition function with parameters:")
            print(f"  - model: {type(model_wrapper)}")
            print(f"  - ref_point: {ref_point_tensor} (shape: {ref_point_tensor.shape})")
            print(f"  - X_baseline: shape {X_baseline.shape}")
            print(f"  - alpha: {alpha_value}")
            
            try:
                acq_func = qNoisyExpectedHypervolumeImprovement(
                    model=model_wrapper,
                    ref_point=ref_point_tensor.tolist(),  # Convert to list for better compatibility
                    X_baseline=X_baseline,
                    prune_baseline=True,
                    alpha=alpha_value,
                    sampler=sampler,  # Use the created sampler
                    cache_root=False,  # Disable caching to avoid potential issues
                )
            except Exception as e:
                print(f"Error creating acquisition function: {e}")
                print("Falling back to random sampling")
                return self._random_candidates(effective_batch_size)
        except Exception as e:
            print(f"Error creating acquisition function: {e}")
            print("Falling back to Thompson Sampling for this batch...")
            
            # Simply return random candidates as the last resort
            print("Using random sampling as final fallback")
            return self._random_candidates(effective_batch_size)
        
        # Optimize acquisition function
        print(f"Optimizing acquisition function to find {effective_batch_size} candidates...")
        
        # Initialize with Sobol samples
        n_samples = 1000
        
        # Use standard bounds [0, 1] for optimization
        standard_bounds = torch.zeros(2, X_baseline.shape[1], dtype=torch.double)
        standard_bounds[1] = 1.0
        
        sobol_samples = draw_sobol_samples(bounds=standard_bounds, n=n_samples, q=effective_batch_size).squeeze(0)
        
        # Optimize from multiple random starting points to avoid local optima
        n_restarts = 5
        raw_samples = 100
        
        try:
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=effective_batch_size,
                num_restarts=n_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            print(f"Acquisition value: {acq_values.item():.6f}")
            
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
                        # Get the available categories
                        categories = config.get('categories', [])
                        if not categories:
                            categories = config.get('values', [])
                        
                        # Ensure we have categories to sample from
                        if not categories:
                            raise ValueError(f"No categories defined for parameter {name}")
                            
                        # Convert to regular Python string to avoid numpy string type issues
                        random_params[name] = str(np.random.choice(categories))
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
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
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
    
    def recommend(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Return current Pareto front"""
        # Find non-dominated points
        with torch.no_grad():
            pareto_mask = botorch.utils.multi_objective.is_non_dominated(self.train_y)
            pareto_y = self.train_y[pareto_mask].cpu().numpy()
            pareto_x = self.train_x[pareto_mask]
        
        # Convert parameters to dictionaries
        pareto_xs = [self.adapter.from_framework_format(x) for x in pareto_x]
        pareto_ys = pareto_y.tolist()
        
        return pareto_xs, pareto_ys
        
    def get_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front"""
        if len(self.train_y) == 0:
            return 0.0
        
        with torch.no_grad():
            bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.train_y)
            volume = bd.compute_hypervolume().item()
            
        return volume 

    def _tensors_to_dicts(self, tensors):
        """Convert multiple tensor candidates to parameter dictionaries"""
        return [self.adapter.from_framework_format(x) for x in tensors] 

    def _random_candidates(self, batch_size):
        """Generate random candidates as fallback"""
        candidates = []
        for _ in range(batch_size):
            random_params = {}
            for name, config in self.parameter_space.parameters.items():
                if config['type'] == 'continuous':
                    random_params[name] = config['bounds'][0] + np.random.random() * (config['bounds'][1] - config['bounds'][0])
                elif config['type'] == 'integer':
                    random_params[name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                elif config['type'] == 'categorical':
                    # Get the available categories
                    categories = config.get('categories', [])
                    if not categories:
                        categories = config.get('values', [])
                    
                    # Ensure we have categories to sample from
                    if not categories:
                        raise ValueError(f"No categories defined for parameter {name}")
                        
                    # Convert to regular Python string to avoid numpy string type issues
                    random_params[name] = str(np.random.choice(categories))
            candidates.append(random_params)
        return candidates 