import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

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
import matplotlib.pyplot as plt
import os

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.botorch_adapter import BoTorchAdapter

class QNEHVI(MultiObjectiveOptimizer):
    """
    q-Noisy Expected Hypervolume Improvement (qNEHVI) implementation using BoTorch
    
    qNEHVI efficiently generates batches of candidates by integrating over the
    unknown function values at previously evaluated designs.
    
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
        **kwargs
    ):
        """
        Initialize the qNEHVI optimizer
        
        Args:
            parameter_space: Parameter space to optimize
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            n_objectives: Number of objectives to optimize
            ref_point: Reference point for hypervolume calculation
            noise_std: Standard deviation of observation noise for each objective
            mc_samples: Number of MC samples for acquisition function approximation
        """
        self.mc_samples = mc_samples
        self.ref_point = ref_point
        self.noise_std = noise_std
        
        # Track model training metrics
        self.model_metrics = {
            'train_losses': [],
            'val_losses': []
        }
        
        super().__init__(parameter_space, budget, batch_size, n_objectives)
        
        # Track timings for progress estimation
        self.timing_history = {
            'model_update': [],
            'acquisition_optimization': [],
            'iteration': []
        }
        
        # Track evaluations and remaining budget
        self.evaluated_points = 0
    
    def _setup(self):
        """Initialize BoTorch components"""
        self.adapter = BoTorchAdapter(self.parameter_space)
        self.bounds = self.adapter.botorch_space['bounds']
        self.train_x = torch.empty((0, len(self.parameter_space.parameters)), dtype=torch.double)
        self.train_y = torch.empty((0, self.n_objectives), dtype=torch.double)
        
        # Set default reference point if not provided
        if self.ref_point is None:
            self.ref_point = torch.ones(self.n_objectives) * 10.0
        else:
            self.ref_point = torch.tensor(self.ref_point, dtype=torch.double)
        
        # Initialize model
        self.model = None
    
    def _update_model(self, output_dir=None):
        """Update the GP model with current training data"""
        start_time = time.time()
        
        if len(self.train_x) < 2:
            # Not enough data to fit a GP
            return None
            
        # Normalize inputs - fix bounds format
        bounds_t = self.bounds.transpose(0, 1)  # Make it (n_dims, 2)
        X = normalize(self.train_x, bounds_t)
        Y = self.train_y
        
        print(f"Fitting GP model with {len(X)} observations...")
        
        # Create and fit a model for each objective
        models = []
        
        for i in tqdm(range(self.n_objectives), desc="Fitting GP models"):
            y = Y[:, i:i+1]  # Get ith objective, keep dimension
            # Create GP model with standardized outputs
            model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            models.append(model)
            
        # Create a ModelListGP from the individual models
        model_list = ModelListGP(*models)
        
        # Store the model for prediction plots
        self.model = model_list
            
        elapsed = time.time() - start_time
        self.timing_history['model_update'].append(elapsed)
        print(f"Model fitting completed in {elapsed:.2f} seconds")
        
        # Create prediction vs true plots if output_dir is provided and we have enough data
        if output_dir is not None and len(self.train_x) >= 5:
            try:
                plots_dir = os.path.join(output_dir, "model_plots")
                iter_num = len(self.timing_history['model_update'])
                # Use perturbation sampling by default to test generalization around existing points
                # And use true evaluation values (not random values)
                self.plot_true_vs_predicted(plots_dir, iter_num, use_random_values=False, sample_method="random")
                print(f"Model prediction plots saved to {plots_dir}")
            except Exception as e:
                print(f"Warning: Could not create model prediction plots: {e}")
                print("Continuing optimization without plots...")
        
        return model_list
    
    def get_model_metrics(self):
        """Return model training metrics (train and validation losses per objective)."""
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

    def plot_and_save_model_error(self, output_dir: str):
        """Plot and save model error (train/val loss) per epoch to output_dir."""
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
        os.makedirs(output_dir, exist_ok=True)
        
        # Make sure we have a model
        if not hasattr(self, 'model') or self.model is None:
            print("Warning: Models not available for plotting predictions")
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

                problem_name = 'complex_categorical'
                print(f"Using test problem '{problem_name}' for evaluating prediction accuracy")
                print(f"Our parameters: {len(best_param_match['our_params'])}, Test problem parameters: {len(best_param_match['problem_params'])}")
                test_problem = get_test_problem(problem_name)
                
                # Evaluate test points using the test problem
                true_values = []
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
                # except Exception as e:
                #     print(f"Error evaluating test points: {e}")
                #     print("Falling back to random values for true evaluations")
                #     true_np = np.random.rand(n_test_points, self.n_objectives) * 10
                    
            except Exception as e:
                print(f"Could not use test problem for evaluation: {e}")
                print("Falling back to random values for true evaluations")
                true_np = np.random.rand(n_test_points, self.n_objectives) * 10
        
        # Get predictions for the test points
        with torch.no_grad():
            X_norm_test = normalize(X_test_tensor, self.bounds.transpose(0, 1))
            predictions_test = self.model.posterior(X_norm_test).mean
            
        # Convert predictions to numpy for easier handling
        pred_test_np = predictions_test.cpu().numpy()
        
        # Get predictions for the training points
        with torch.no_grad():
            X_norm_train = normalize(self.train_x, self.bounds.transpose(0, 1))
            predictions_train = self.model.posterior(X_norm_train).mean
        
        # Convert predictions and true values to numpy
        pred_train_np = predictions_train.cpu().numpy()
        true_train_np = self.train_y.cpu().numpy()
        
        # Plot for each objective
        for i in range(self.n_objectives):
            plt.figure(figsize=(12, 8))
            
            # Plot training data (seen)
            plt.scatter(true_train_np[:, i], pred_train_np[:, i], alpha=0.7, color='blue', 
                       label=f'Training data (n={len(true_train_np)})', marker='o')
            
            # Plot test data (unseen)
            plt.scatter(true_np[:, i], pred_test_np[:, i], alpha=0.5, color='red', 
                       label=f'Test data (n={len(true_np)})', marker='x')
            
            # Find global min and max for perfect prediction line
            all_true = np.concatenate([true_train_np[:, i], true_np[:, i]])
            all_pred = np.concatenate([pred_train_np[:, i], pred_test_np[:, i]])
            min_val = min(np.min(all_true), np.min(all_pred))
            max_val = max(np.max(all_true), np.max(all_pred))
            
            # Plot perfect prediction line
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
            
            # Calculate error statistics for test data
            test_mse = np.mean((true_np[:, i] - pred_test_np[:, i])**2)
            test_mae = np.mean(np.abs(true_np[:, i] - pred_test_np[:, i]))
            
            # Calculate error statistics for training data
            train_mse = np.mean((true_train_np[:, i] - pred_train_np[:, i])**2)
            train_mae = np.mean(np.abs(true_train_np[:, i] - pred_train_np[:, i]))
            
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.title(f'True vs Predicted - Objective {i+1}\n' +
                     f'Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}\n' +
                     f'Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')
            plt.legend()
            plt.grid(True)
            
            # Add error statistics as text
            stats_text = (f"Training errors: MSE={train_mse:.4f}, MAE={train_mae:.4f}\n"
                         f"Test errors: MSE={test_mse:.4f}, MAE={test_mae:.4f}")
            plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f'true_vs_pred_obj_{i+1}_iter_{iter_num}.png'), dpi=150)
            plt.close() 