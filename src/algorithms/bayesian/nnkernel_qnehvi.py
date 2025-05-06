import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import gpytorch
import torch.nn as nn
import torch.nn.functional as F
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
from src.utils.variance_calculator import VarianceCalculator

# Define the neural network model for mean prediction
class MLP(nn.Module):
    """Multi-layer perceptron for feature mapping in kernel space"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], hidden_map_dim: Optional[List[int]] = None, 
                 output_dim: int = 1, dtype=torch.float64, dropout_rate: float = 0.2):
        """
        Initialize MLP network
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            hidden_map_dim: Dimensions for the hidden map output (if not None)
            output_dim: Output dimension (typically 1)
            dtype: Data type to use for the model
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_map_dim = hidden_map_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        
        # Construct a simpler network with fewer layers and less regularization
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            # Linear layer
            layer = nn.Linear(prev_dim, dim, dtype=dtype)
            layers.append(layer)
            
            # Initialize with small weights but non-zero (orthogonal initialization)
            nn.init.orthogonal_(layer.weight, gain=0.8)
            nn.init.zeros_(layer.bias)
            
            # Activation - use TanH instead of LeakyReLU for better stability and bounded outputs
            layers.append(nn.Tanh())
            
            # Use less dropout
            if i < len(hidden_dims) - 1:  # No dropout before final output
                layers.append(nn.Dropout(dropout_rate * 0.5))
            
            prev_dim = dim
        
        # Final output layer for embedding
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim, dtype=dtype)
        nn.init.orthogonal_(self.output_layer.weight, gain=0.8)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        # Ensure input has the correct dtype
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)
        
        # Forward through feature extractor
        features = self.feature_extractor(x)
        
        # Through output layer
        output = self.output_layer(features)
        
        return output


# Add this after MLP class and before NeuralNetworkMean class

class PhiKernel(gpytorch.kernels.Kernel):
    """
    Kernel that uses a pretrained phi network to transform inputs before computing distances.
    
    This allows for complex, non-linear feature mapping in kernel space.
    """
    
    def __init__(self, phi_network, base_kernel=None, jitter=1e-4, **kwargs):
        """Initialize with phi network for feature transformation"""
        super().__init__(**kwargs)
        self.phi_network = phi_network
        self.jitter = jitter
        
        # Use RBF as default base kernel if none provided
        if base_kernel is None:
            self.base_kernel = gpytorch.kernels.RBFKernel()
        else:
            self.base_kernel = base_kernel
    
    def forward(self, x1, x2=None, diag=False, **params):
        """Apply phi transformation then compute kernel"""
        # Project inputs through phi network
        with torch.no_grad():
            if x1.dtype != next(self.phi_network.parameters()).dtype:
                x1 = x1.to(dtype=next(self.phi_network.parameters()).dtype)
            
            # Get embeddings without normalization to preserve the learned structure
            phi_x1 = self.phi_network(x1)
            
            # Minimal scaling to ensure numerical stability without changing relationships
            # Avoid mean subtraction which affects the relative positions of points
            phi_x1_std = torch.std(phi_x1, dim=0, keepdim=True)
            phi_x1_std = torch.clamp(phi_x1_std, min=1e-8)  # Avoid division by zero
            phi_x1 = phi_x1 / phi_x1_std  # Scale only, no centering
            
            if x2 is not None:
                if x2.dtype != next(self.phi_network.parameters()).dtype:
                    x2 = x2.to(dtype=next(self.phi_network.parameters()).dtype)
                phi_x2 = self.phi_network(x2)
                
                # Apply same scaling to x2 using x1's stats for consistency
                phi_x2 = phi_x2 / phi_x1_std
            else:
                phi_x2 = None
        
        # Use base kernel on transformed features
        base_kernel_output = self.base_kernel(phi_x1, phi_x2, diag=diag, **params)
        
        # If we're computing diagonal elements only, no need to modify further
        if diag:
            return base_kernel_output
        
        # If the output is a MultivariateNormal distribution, return as is
        if isinstance(base_kernel_output, torch.distributions.MultivariateNormal):
            return base_kernel_output
            
        # For matrix outputs, process and add jitter
        if x2 is None:
            # Add jitter when computing self-covariance
            if hasattr(base_kernel_output, 'evaluate'):
                # This is a lazy tensor
                K = base_kernel_output.evaluate()
            else:
                # This is already a tensor
                K = base_kernel_output
                
            # Add jitter only to the diagonal for numerical stability
            eye_matrix = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
            return K + self.jitter * eye_matrix
        
        # If x1 ≠ x2, no jitter needed
        return base_kernel_output




# Create custom GP model with neural network kernel
class NeuralNetworkGP(SingleTaskGP):
    """
    Gaussian Process with neural network kernel.
    
    This hybrid model uses neural networks for kernel feature mapping
    while maintaining standard GP mean functions.
    """
    
    def __init__(
        self, 
        train_X: torch.Tensor, 
        train_Y: torch.Tensor, 
        phi_network: nn.Module,
        base_kernel = None,
        mean_module = None,
        likelihood = None,
        outcome_transform = None,
        jitter = 1e-3
    ):
        """
        Initialize the neural network kernel GP model
        
        Args:
            train_X: Training inputs
            train_Y: Training targets
            phi_network: Neural network model for kernel feature mapping
            base_kernel: Base kernel to use with phi features (default: RBF)
            mean_module: Optional custom mean module (default: ConstantMean)
            likelihood: GPyTorch likelihood
            outcome_transform: Outcome transform
            jitter: Jitter to add to kernel for numerical stability
        """
        # Print input shapes for debugging
        print(f"NeuralNetworkGP init: train_X shape={train_X.shape}, train_Y shape={train_Y.shape}")
        
        # Create a default likelihood if none provided
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
                noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)  # Prior encouraging small noise values
            )
            
        # Use default mean module if none provided
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
            
        # Create neural network enhanced kernel
        covar_module = PhiKernel(phi_network, base_kernel=base_kernel, jitter=jitter)
        
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
            
        # Initialize the parent GP model with default mean and neural network kernel
        super().__init__(
            train_X=train_X, 
            train_Y=train_Y, 
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            outcome_transform=outcome_transform
        )
        
        # Set initial noise to a small value to encourage exact fitting
        self.likelihood.noise = torch.tensor(0.01)
        
        # Store neural network model
        self.phi_network = phi_network




class NNKernelQNEHVI(QNEHVI):
    """
    Neural Network Kernel enhanced q-Noisy Expected Hypervolume Improvement (qNEHVI)
    
    This implementation uses neural networks to map inputs into a feature space,
    creating a custom kernel for GP modeling. The neural network learns to project
    inputs into a space where the standard kernel (e.g., RBF) can better capture
    complex relationships between inputs and outputs.
    
    The approach is similar to deep kernel learning, where we:
    1. Train neural networks to learn meaningful feature mappings
    2. Use these networks as input transformations for GP kernels
    3. Optimize hyperparameters for both the neural network and GP
    
    References:
    [1] S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of 
        Multiple Noisy Objectives with Expected Hypervolume Improvement. Advances 
        in Neural Information Processing Systems 34, 2021.
    [2] Wilson, A.G., Hu, Z., Salakhutdinov, R., & Xing, E.P. (2016). 
        Deep Kernel Learning. Artificial Intelligence and Statistics.
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
        
        nn_layers: List[int] = [20, 16],
        nn_learning_rate: float = 0.01,
        nn_epochs: int = 300,
        nn_batch_size: int = 16,
        nn_regularization: float = 1e-4,
        hidden_map_dim: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize the Neural Network kernel qNEHVI optimizer
        
        Args:
            parameter_space: Parameter space to optimize
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            n_objectives: Number of objectives to optimize
            ref_point: Reference point for hypervolume calculation
            noise_std: Standard deviation of observation noise for each objective
            mc_samples: Number of MC samples for acquisition function approximation
            nn_layers: Hidden layer sizes for the neural network feature mapping
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
        
        # Initialize neural network models for kernel feature mapping
        self.phi_models = []
        
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
        
        # Use smaller embedding dimension
        embedding_dim = 8
        
        # Create neural network model with the same dtype as the input data
        model = MLP(
            input_dim=input_dim, 
            hidden_dims=self.nn_layers, 
            hidden_map_dim=self.hidden_map_dim if hidden_maps is not None else None,
            dtype=dtype,
            dropout_rate=0.2,  # Reduced dropout
            output_dim=embedding_dim  # Smaller embedding dimension
        )
        
        # Use MSE loss for main prediction
        mse_criterion = nn.MSELoss()
        
        # Define optimizer with appropriate learning rate and less regularization
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.nn_learning_rate,  # Use full learning rate
            weight_decay=self.nn_regularization * 0.5  # Less L2 regularization
        )
        
        # Add learning rate scheduler with more patience
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=30,
            verbose=True, min_lr=1e-6
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
        
        # Apply minimal noise to training data for better generalization
        if X_train.shape[0] > 0:
            noise_scale = 1e-4  # Much less noise
            X_train_noisy = X_train + torch.randn_like(X_train) * noise_scale
        else:
            X_train_noisy = X_train
        
        # Create data loader for mini-batch training
        if hidden_maps is not None:
            train_dataset = torch.utils.data.TensorDataset(X_train_noisy, y_train, hidden_maps_train)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train_noisy, y_train)
        
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
        
        # Early stopping parameters - more patience to allow convergence
        patience = 50
        patience_counter = 0
        
        # Define map loss weight - adjust this to control importance
        map_loss_weight = 0.01
        
        # Use tqdm for progress tracking
        for epoch in tqdm(range(self.nn_epochs), desc=f"NN training (obj {objective_idx+1})"):
            # Training phase
            model.train()
            train_loss = 0.0
            
            # Handle different batch structures based on hidden map presence
            if hidden_maps is not None:
                # Create VarianceCalculator for map distance calculation
                VC = VarianceCalculator()
                
                for batch_X, batch_y, batch_maps in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass - returns outputs (embeddings)
                    outputs = model(batch_X)
                    
                    # Calculate pairwise distance matrix between maps in the batch
                    # Convert PyTorch tensors to numpy arrays for VarianceCalculator
                    batch_maps_np = batch_maps.detach().cpu().numpy()
                    
                    # Need to reshape the maps if needed (assuming batch_maps is [batch_size, map_h, map_w])
                    # Each map should be a 2D grid for the 2D Wasserstein distance calculation
                    maps_list = [batch_maps_np[i] for i in range(batch_maps_np.shape[0])]
                    
                    # Calculate pairwise distances between maps
                    # This returns a distance matrix of shape [batch_size, batch_size]
                    map_distance_matrix = VC.calculate_pairwise_2d_distribution_distance(
                        maps_list, 
                        method= 'sliced', # 'sliced', 'mmd_1d'
                        num_projections=20,
                        target_size=None
                    )['distance_matrix']
                    # Convert distance matrix to similarity matrix in PyTorch
                    # Invert and normalize distances to similarities (closer = more similar)
                    map_distance_tensor = torch.tensor(map_distance_matrix, dtype=dtype)
                    
                    # Handle potential NaN values in the distance matrix
                    map_distance_tensor = torch.nan_to_num(map_distance_tensor, nan=0.0)
                    
                    # Calculate the embedding similarity matrix using cosine similarity
                    # Cosine is more stable than dot product for similarity
                    normalized_embeddings = F.normalize(outputs, p=2, dim=1)
                    embedding_similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())
                    
                    # Normalize map distances (smaller distance = higher similarity)
                    # Add small constant to avoid division by zero
                    eps = 1e-8
                    map_max_dist = torch.max(map_distance_tensor) + eps
                    map_similarity = 1.0 - (map_distance_tensor / map_max_dist)
                    
                    # Calculate similarity loss (MSE between the two similarity matrices)
                    loss = F.mse_loss(embedding_similarity, map_similarity.to(embedding_similarity.device, dtype=embedding_similarity.dtype))
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                if hidden_maps is not None:
                    # Extract validation maps
                    hidden_maps_val_np = hidden_maps_val.detach().cpu().numpy()
                    maps_list_val = [hidden_maps_val_np[i] for i in range(hidden_maps_val_np.shape[0])]
                    
                    # Calculate map distance matrix for validation data
                    val_map_distance_matrix = VC.calculate_pairwise_2d_distribution_distance(
                        maps_list_val,
                        method='sliced',
                        num_projections=50,
                        target_size=50
                    )['distance_matrix']
                    
                    # Convert to similarity
                    val_map_distance_tensor = torch.tensor(val_map_distance_matrix, dtype=dtype)
                    # Handle potential NaN values
                    val_map_distance_tensor = torch.nan_to_num(val_map_distance_tensor, nan=0.0)
                    
                    # Normalize map distances (smaller distance = higher similarity)
                    eps = 1e-8
                    val_map_max_dist = torch.max(val_map_distance_tensor) + eps
                    val_map_similarity = 1.0 - (val_map_distance_tensor / val_map_max_dist)
                    
                    # Forward pass to get outputs
                    val_outputs = model(X_val)
                    
                    # Calculate embedding similarity
                    val_normalized_embeddings = F.normalize(val_outputs, p=2, dim=1)
                    val_embedding_similarity = torch.mm(val_normalized_embeddings, val_normalized_embeddings.t())
                    
                    # Calculate similarity loss - ensure dtype consistency
                    val_loss = F.mse_loss(
                        val_embedding_similarity, 
                        val_map_similarity.to(val_embedding_similarity.device, dtype=val_embedding_similarity.dtype)
                    )
                else:
                    val_outputs = model(X_val)
                    # Ensure consistent dtype
                    val_loss = mse_criterion(val_outputs, y_val.to(val_outputs.dtype))
                
                val_losses.append(val_loss)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Stop if no improvement for a while
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress periodically
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if hidden_maps is not None:
                    print(f"Epoch {epoch+1}/{self.nn_epochs}, Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
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
        
        print(f"Fitting NN-kernel GP model with {len(X)} observations...")
        print(f"Input tensor shapes: X={X.shape}, Y={Y.shape}")
        # Create and fit a model for each objective
        models = []
        self.phi_models = []  # Reset neural network models
        
        for i in tqdm(range(self.n_objectives), desc="Fitting NN-kernel models"):
            y = Y[:, i]  # Get ith objective, keep as 1D
            print(f"Objective {i+1}: y shape = {y.shape}, y dtype = {y.dtype}")
            
            # Train neural network for kernel feature mapping
            input_dim = X.shape[1]
            if self.train_hidden_maps is not None:
                phi_model = self._train_neural_network(X, y.unsqueeze(-1), input_dim, i, self.train_hidden_maps)  # Pass 2D for training
            else:
                phi_model = self._train_neural_network(X, y.unsqueeze(-1), input_dim, i)  # Pass 2D for training
            self.phi_models.append(phi_model)
            
            # Create a combined kernel for better stability: Matern kernel is more numerically stable than RBF
            # Use a lower nu value for a more flexible kernel
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=phi_model.output_dim)
            
            # Create GP model with neural network kernel
            model = NeuralNetworkGP(
                train_X=X,
                train_Y=y,  # Pass 1D tensor as BoTorch expects
                phi_network=phi_model,
                # Use Matern kernel as base with larger jitter 
                base_kernel=base_kernel,
                # Add noise likelihood with reasonable constraint
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-5)
                ),
                outcome_transform=Standardize(m=1)
            )
            
            # Set initial noise to a small value to encourage fitting data closely
            model.likelihood.noise = torch.tensor(0.01)
            
            # Fit GP parameters (keeping neural network fixed)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            # Only optimize GP hyperparameters
            try:
                # Configure optimizer settings for better convergence
                fit_gpytorch_mll(
                    mll,
                    options={
                        "max_iter": 200,    # More iterations
                        "lr": 0.1,          # Higher learning rate
                        "disp": True        # Show progress
                    }
                )
                
                # Check if model has reasonable noise parameter
                print(f"Fitted noise parameter: {model.likelihood.noise.item()}")
                
                # If noise is too high, try to fit again with a stronger noise constraint
                if model.likelihood.noise.item() > 0.1:
                    print("Noise parameter too large, refitting with stronger constraint")
                    model.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.Interval(1e-5, 0.05)
                    )
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                
                # Verify kernel works properly
                with torch.no_grad():
                    K = model.covar_module(X).evaluate()
                    print(f"Kernel matrix shape: {K.shape}")
                    print(f"Kernel matrix stats: min={K.min().item():.4f}, max={K.max().item():.4f}, mean={K.mean().item():.4f}")
                    
                    # Check if kernel matrix has NaN values
                    if torch.any(torch.isnan(K)):
                        print("Warning: NaN values in kernel matrix")
            
            except Exception as e:
                print(f"Error fitting GP: {e}")
                print("Continuing with potentially suboptimal model")
            
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
        
        Args:
            output_dir: Directory to save plots
            iter_num: Iteration number for naming the plot
            use_random_values: If False (default), use actual test problem evaluations for true values
            sample_method: Method to sample test points - options: "sobol", "random", "perturb", "acquisition"
        """
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Make sure we have a model
        if not hasattr(self, 'model') or self.model is None:
            print("Warning: GP models not available for plotting predictions")
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
        
        # Normalize inputs for GP prediction
        X_norm_test = normalize(X_test_tensor, self.bounds.transpose(0, 1))
        
        # Get GP predictions for the test points
        with torch.no_grad():
            gp_predictions_test = self.model.posterior(X_norm_test).mean
            
        # Convert predictions to numpy for easier handling
        gp_pred_test_np = gp_predictions_test.cpu().numpy()
        
        # Get predictions for the training points
        with torch.no_grad():
            X_norm_train = normalize(self.train_x, self.bounds.transpose(0, 1))
            gp_predictions_train = self.model.posterior(X_norm_train).mean
            
        # Convert training predictions and true values to numpy
        gp_pred_train_np = gp_predictions_train.cpu().numpy()
        true_train_np = self.train_y.cpu().numpy()
        
        # Plot for each objective
        for i in range(self.n_objectives):
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.title(f'GP Predictions - Objective {i+1}')
            
            # Plot training data points
            plt.scatter(true_train_np[:, i], gp_pred_train_np[:, i], alpha=0.7, color='blue',
                       marker='o', label='Training Data')
            
            # Plot test data points
            plt.scatter(true_np[:, i], gp_pred_test_np[:, i], alpha=0.7, color='red',
                       marker='x', label='Test Data')
            
            # Find global min and max for diagonal line
            global_min = min(np.min(true_train_np[:, i]), np.min(true_np[:, i]),
                          np.min(gp_pred_train_np[:, i]), np.min(gp_pred_test_np[:, i]))
            global_max = max(np.max(true_train_np[:, i]), np.max(true_np[:, i]),
                          np.max(gp_pred_train_np[:, i]), np.max(gp_pred_test_np[:, i]))
            plt.plot([global_min, global_max], [global_min, global_max], 'k--', label='Perfect prediction')
            
            plt.xlabel('True values')
            plt.ylabel('GP predicted values')
            plt.legend()
            
            # Calculate error metrics
            gp_train_mse = np.mean((true_train_np[:, i] - gp_pred_train_np[:, i])**2)
            gp_train_mae = np.mean(np.abs(true_train_np[:, i] - gp_pred_train_np[:, i]))
            gp_test_mse = np.mean((true_np[:, i] - gp_pred_test_np[:, i])**2)
            gp_test_mae = np.mean(np.abs(true_np[:, i] - gp_pred_test_np[:, i]))
            
            # Add error metrics text box
            plt.text(0.05, 0.95,
                   f'Train - MSE: {gp_train_mse:.4f}, MAE: {gp_train_mae:.4f}\n' +
                   f'Test - MSE: {gp_test_mse:.4f}, MAE: {gp_test_mae:.4f}',
                   transform=plt.gca().transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'gp_predictions_obj_{i+1}_iter_{iter_num}.png'), dpi=150)
            plt.close()
            
            # Also create a separate plot for training and test data
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training data subplot
            ax1.set_title(f'Training Data - Objective {i+1}')
            ax1.scatter(true_train_np[:, i], gp_pred_train_np[:, i], alpha=0.7, color='blue', marker='o')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_train_np[:, i]), np.min(gp_pred_train_np[:, i]))
            max_val = max(np.max(true_train_np[:, i]), np.max(gp_pred_train_np[:, i]))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            ax1.set_xlabel('True values')
            ax1.set_ylabel('GP predicted values')
            
            # Add error metrics
            ax1.text(0.05, 0.95, f'MSE: {gp_train_mse:.4f}\nMAE: {gp_train_mae:.4f}',
                   transform=ax1.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Test data subplot
            ax2.set_title(f'Test Data - Objective {i+1}')
            ax2.scatter(true_np[:, i], gp_pred_test_np[:, i], alpha=0.7, color='red', marker='x')
            
            # Find min and max for diagonal line
            min_val = min(np.min(true_np[:, i]), np.min(gp_pred_test_np[:, i]))
            max_val = max(np.max(true_np[:, i]), np.max(gp_pred_test_np[:, i]))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            ax2.set_xlabel('True values')
            ax2.set_ylabel('GP predicted values')
            
            # Add error metrics
            ax2.text(0.05, 0.95, f'MSE: {gp_test_mse:.4f}\nMAE: {gp_test_mae:.4f}',
                   transform=ax2.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Add overall title
            plt.suptitle(f'GP Prediction Analysis - Objective {i+1}')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for suptitle
            
            # Save the split plot
            plt.savefig(os.path.join(output_dir, f'gp_train_test_obj_{i+1}_iter_{iter_num}.png'), dpi=150)
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
            # plt.savefig(os.path.join(output_dir, f'model_error_objective_{i+1}.png'))
            plt.close() 