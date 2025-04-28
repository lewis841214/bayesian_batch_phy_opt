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
    
    def _update_model(self):
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
            model = SingleTaskGP(X, y, outcome_transform=None)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            models.append(model)
            
        # Create a ModelListGP from the individual models
        model_list = ModelListGP(*models)
            
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
            
            # Create the acquisition function with the sampler
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=self.ref_point.clone().detach(),  # Proper tensor cloning
                X_baseline=X_baseline,
                prune_baseline=True,
                alpha=alpha,
                sampler=sampler,  # Use the created sampler
            )
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
        
        # Initialize with Sobol samples
        # n_samples = 1000
        
        # Use standard bounds [0, 1] for optimization
        standard_bounds = torch.zeros(2, X_baseline.shape[1], dtype=torch.double)
        standard_bounds[1] = 1.0
        
        # sobol_samples = draw_sobol_samples(bounds=standard_bounds, n=n_samples, q=effective_batch_size).squeeze(0)
        
        # Optimize from multiple random starting points to avoid local optima
        n_restarts = 50
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