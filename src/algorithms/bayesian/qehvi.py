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
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from pymoo.indicators.hv import HV
import time
from tqdm import tqdm

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.botorch_adapter import BoTorchAdapter

class QEHVI(MultiObjectiveOptimizer):
    """
    q-Expected Hypervolume Improvement (qEHVI) implementation using BoTorch
    
    qEHVI maximizes the expected increase in the dominated hypervolume. This
    implementation uses FastNondominatedPartitioning to compute the improvement.
    
    References:
    [1] S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume 
        Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in 
        Neural Information Processing Systems 33, 2020.
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
        Initialize the qEHVI optimizer
        
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
        """Update the model with current training data"""
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
            
            # Configure noise if known
            if self.noise_std is not None and isinstance(self.noise_std, list):
                noise_var = torch.full_like(y, self.noise_std[i] ** 2)
                model = SingleTaskGP(X, y, noise_var, outcome_transform=Standardize(m=1))
            else:
                model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
                
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            models.append(model)
        
        # Create a ModelListGP
        model_list = ModelListGP(*models)
            
        elapsed = time.time() - start_time
        self.timing_history['model_update'].append(elapsed)
        print(f"Model fitting completed in {elapsed:.2f} seconds")
        
        return model_list
    
    def ask(self):
        """Return a batch of points to evaluate"""
        # If we don't have enough points, generate random samples
        if len(self.train_x) < 2 * self.n_objectives:
            print("Insufficient data for GP model. Using random sampling.")
            
            # Generate candidates directly using the parameter space
            candidates = []
            for _ in range(self.batch_size):
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
                        random_params[name] = np.random.choice(config['categories'])
                candidates.append(random_params)
                
            return candidates
        
        # Update GP models
        model_list = self._update_model()
        
        if model_list is None:
            # Fall back to random sampling
            candidates = []
            for _ in range(self.batch_size):
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
                        random_params[name] = np.random.choice(config['values'])
                candidates.append(random_params)
                
            return candidates
        
        start_time = time.time()
        
        # Create the acquisition function
        print("Creating qEHVI acquisition function...")
        
        # Normalize inputs for acquisition function
        bounds_t = self.bounds.transpose(0, 1)
        X_baseline = normalize(self.train_x, bounds_t)
        
        try:
            # Get model predictions at training points
            with torch.no_grad():
                pred = model_list.posterior(X_baseline).mean
            
            # Partition non-dominated space using FastNondominatedPartitioning
            partitioning = FastNondominatedPartitioning(
                ref_point=self.ref_point.clone().detach(),
                Y=pred,
            )
            
            # Create sampler for MC integration
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
            
            # Create qEHVI acquisition function
            acq_func = qExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=self.ref_point.clone().detach(),
                partitioning=partitioning,
                sampler=sampler,
            )
            
            # Optimize acquisition function
            print(f"Optimizing acquisition function to find {self.batch_size} candidates...")
            
            # Use standard bounds [0, 1] for optimization
            standard_bounds = torch.zeros(2, X_baseline.shape[1], dtype=torch.double)
            standard_bounds[1] = 1.0
            
            # Initialize with Sobol samples for better starting points
            n_samples = 1000
            sobol_samples = draw_sobol_samples(bounds=standard_bounds, n=n_samples, q=self.batch_size).squeeze(0)
            
            # Optimize from multiple random starting points to avoid local optima
            n_restarts = 5
            raw_samples = 100
            
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=self.batch_size,
                num_restarts=n_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            print(f"Acquisition value: {acq_values.item():.6f}")
            
            # Unnormalize candidates
            candidates = unnormalize(candidates.detach(), bounds_t)
            
        except Exception as e:
            print(f"Error in acquisition function optimization: {str(e)}")
            # Fall back to random sampling or use most recent points
            candidates = self.train_x[-self.batch_size:]
        
        elapsed = time.time() - start_time
        self.timing_history['acquisition_optimization'].append(elapsed)
        print(f"Acquisition optimization completed in {elapsed:.2f} seconds")
        
        # Convert tensor to dictionaries
        candidate_dicts = self._tensors_to_dicts(candidates)
        
        # Return a batch of candidates
        return candidate_dicts
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update model with evaluated points"""
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        self.timing_history['iteration'].append(elapsed)
        
        # Provide timing estimates for next iteration
        avg_model_time = 0.0
        avg_acq_time = 0.0
        avg_iter_time = 0.0
        
        if len(self.timing_history['model_update']) > 0:
            avg_model_time = np.mean(self.timing_history['model_update'])
            print(f"Average model update time: {avg_model_time:.2f} seconds")
            
        if len(self.timing_history['acquisition_optimization']) > 0:
            avg_acq_time = np.mean(self.timing_history['acquisition_optimization'])
            print(f"Average acquisition optimization time: {avg_acq_time:.2f} seconds")
            
        if len(self.timing_history['iteration']) > 0:
            avg_iter_time = np.mean(self.timing_history['iteration'])
            print(f"Average iteration time: {avg_iter_time:.2f} seconds")
            
        # Estimate remaining time
        remaining_steps = (self.budget - len(self.train_x)) / self.batch_size
        est_remaining_time = remaining_steps * (avg_model_time + avg_acq_time + avg_iter_time)
        print(f"Estimated remaining time: {est_remaining_time:.2f} seconds ({est_remaining_time/60:.2f} minutes)")
    
    def _tensors_to_dicts(self, tensors):
        """Convert multiple tensor candidates to parameter dictionaries"""
        return [self.adapter.from_framework_format(x) for x in tensors]
    
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