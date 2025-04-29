import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import time

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from pymoo.indicators.hv import HV


class RandomSearch(MultiObjectiveOptimizer):
    """
    Simple random search algorithm for multi-objective optimization.
    
    This serves as a baseline for comparison with more sophisticated algorithms
    like Bayesian optimization.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        budget: int,
        batch_size: int = 1, 
        n_objectives: int = 2,
        ref_point: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize the random search optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            n_objectives: Number of objectives to optimize
            ref_point: Reference point for hypervolume calculation
        """
        self.evaluated_xs = []
        self.evaluated_ys = []
        self.evaluations_count = 0
        
        super().__init__(parameter_space, budget, batch_size, n_objectives)
        
        # Set default reference point if not provided
        if ref_point is None:
            self.ref_point = np.array([100.0] * self.n_objectives)
        else:
            self.ref_point = np.array(ref_point)
    
    def _setup(self):
        """Initialize components for random search"""
        # Nothing special to set up for random search
        pass
    
    def ask(self) -> List[Dict[str, Any]]:
        """Return a batch of random points to evaluate"""
        # Calculate remaining budget
        remaining = self.budget - self.evaluations_count
        effective_batch_size = min(self.batch_size, remaining)
        
        if effective_batch_size <= 0:
            print("Budget exhausted, no more evaluations possible.")
            return []
            
        print(f"Generating random batch of {effective_batch_size} candidates. Remaining budget: {remaining}")
        
        # Generate random points
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
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update with evaluated points"""
        # Store evaluations
        self.evaluated_xs.extend(xs)
        self.evaluated_ys.extend(ys)
        self.evaluations_count += len(xs)
        
        print(f"Tell called with {len(xs)} points. Total evaluated: {self.evaluations_count}/{self.budget}")
    
    def recommend(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Return current Pareto front"""
        if len(self.evaluated_ys) == 0:
            return [], []
        
        # Use manual approach to find non-dominated points for MAXIMIZATION problem
        # For maximization, a point dominates another if it's greater in all objectives
        # with at least one objective being strictly greater
        pareto_indices = []
        for i, y_i in enumerate(self.evaluated_ys):
            is_dominated = False
            for j, y_j in enumerate(self.evaluated_ys):
                if i != j and all(y_j[k] >= y_i[k] for k in range(len(y_i))) and any(y_j[k] > y_i[k] for k in range(len(y_i))):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        
        pareto_xs = [self.evaluated_xs[i] for i in pareto_indices]
        pareto_ys = [self.evaluated_ys[i] for i in pareto_indices]
        
        return pareto_xs, pareto_ys
    
    def get_hypervolume(self) -> float:
        """Calculate hypervolume of current Pareto front"""
        if len(self.evaluated_ys) == 0:
            return 0.0
        
        # Get Pareto front
        _, pareto_ys = self.recommend()
        
        if len(pareto_ys) == 0:
            return 0.0
        
        try:
            # For maximization problems, we need to negate the values for the HV calculator
            # as pymoo's HV implementation assumes minimization
            pareto_array = -1 * np.array(pareto_ys)
            ref_point = -1 * self.ref_point
            
            # Calculate hypervolume using pymoo's HV calculator
            hv_calculator = HV(ref_point=ref_point)
            hv_value = hv_calculator.do(pareto_array)
            return hv_value
        except Exception as e:
            print(f"Error in hypervolume calculation: {e}. Returning 0.")
            return 0.0 