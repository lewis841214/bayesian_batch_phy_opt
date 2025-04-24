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
        if len(self.train_x) == 0:
            return
        
        # Normalize inputs to [0, 1]
        bounds = self.bounds.transpose(0, 1)
        train_x_normalized = normalize(self.train_x, bounds)
        
        # Initialize models for each objective
        models = []
        for i in range(self.n_objectives):
            train_y_i = self.train_y[:, i:i+1]
            
            # Configure noise if known
            if self.noise_std is not None:
                train_yvar = torch.full_like(train_y_i, self.noise_std[i] ** 2)
                model = SingleTaskGP(
                    train_x_normalized, 
                    train_y_i, 
                    train_yvar,
                    outcome_transform=Standardize(m=1)
                )
            else:
                model = SingleTaskGP(
                    train_x_normalized, 
                    train_y_i,
                    outcome_transform=Standardize(m=1)
                )
            
            models.append(model)
        
        # Create a model list and fit
        self.model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
    
    def ask(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return batch of points to evaluate using qNEHVI acquisition"""
        n = n or self.batch_size
        
        # If we don't have enough data to fit a model, sample randomly
        if len(self.train_x) < 2 * self.n_objectives:
            # Generate Sobol samples in the parameter space
            sobol = torch.quasirandom.SobolEngine(dimension=len(self.parameter_space.parameters))
            samples = sobol.draw(n)
            # Scale to bounds
            bounds = self.bounds.transpose(0, 1)
            samples = bounds[0] + (bounds[1] - bounds[0]) * samples
            
            # Convert to parameter dictionaries
            return [self.adapter.from_framework_format(x) for x in samples]
        
        # Update model with current data
        self._update_model()
        
        # Normalize inputs
        bounds = self.bounds.transpose(0, 1)
        train_x_normalized = normalize(self.train_x, bounds)
        
        # Create sampler for MC integration
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
        
        try:
            # Create qNEHVI acquisition function
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=self.model,
                ref_point=self.ref_point.tolist(),
                X_baseline=train_x_normalized,
                prune_baseline=True,  # prune baseline points with near-zero probability of being Pareto optimal
                sampler=sampler,
            )
            
            # Create standard [0,1] bounds for optimization
            standard_bounds = torch.zeros(2, train_x_normalized.shape[1], dtype=torch.double)
            standard_bounds[1] = 1.0
            
            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=n,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            
            # Unnormalize candidates
            candidates = unnormalize(candidates.detach(), bounds)
            
        except Exception as e:
            print(f"Error in acquisition function optimization: {str(e)}")
            # Fall back to random sampling
            sobol = torch.quasirandom.SobolEngine(dimension=len(self.parameter_space.parameters))
            samples = sobol.draw(n)
            candidates = bounds[0] + (bounds[1] - bounds[0]) * samples
        
        # Convert to parameter dictionaries
        return [self.adapter.from_framework_format(x) for x in candidates]
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update model with evaluated points"""
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