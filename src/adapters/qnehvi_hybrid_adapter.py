from typing import Dict, List, Any, Tuple, Optional

from src.core.algorithm import MultiObjectiveOptimizer
from tests.test_algorithms.test_problems import TestProblem
from src.adapters.algorithm_adapters import AlgorithmAdapter
from src.algorithms.bayesian.qnehvi_hybrid import QNEHVIHybrid
from src.core.parameter_space import ParameterSpace

class QNEHVIHybridAdapter(AlgorithmAdapter):
    """Adapter for the QNEHVIHybrid algorithm"""
    
    def __init__(self, surrogate_model="nn"):
        """Initialize adapter
        
        Args:
            surrogate_model: Type of surrogate model to use ("nn" or "xgboost")
        """
        self.surrogate_model = surrogate_model
        self.algorithm = None
        self.problem = None
    
    def setup(self, problem: TestProblem, budget: int, batch_size: int = 1):
        """Set up the algorithm for a given problem
        
        Args:
            problem: Test problem to optimize
            budget: Number of function evaluations
            batch_size: Number of points to evaluate in parallel
        """
        self.problem = problem
        parameter_space = problem.get_parameter_space()
        
        # Set up reference point for hypervolume calculation
        if hasattr(problem, 'get_reference_point'):
            ref_point = problem.get_reference_point()
        else:
            # Default reference point based on problem type
            ref_point = [-float('inf')] * problem.num_objectives
            
        # Create algorithm instance
        self.algorithm = QNEHVIHybrid(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            surrogate_model=self.surrogate_model,
            ref_point=ref_point,
            # Additional hyperparameters can be tuned here
            nn_hidden_dim=64,
            nn_dropout_rate=0.2,
            nn_ensemble_size=5,
            nn_epochs=100,
            xgb_n_estimators=100,
            xgb_max_depth=4,
            xgb_learning_rate=0.1
        )
    
    def ask(self) -> List[Dict[str, Any]]:
        """Get next batch of points to evaluate"""
        candidates = self.algorithm.ask()
        
        # Print candidates for debugging
        print(f"Raw candidates from algorithm: {candidates}")
        
        # Use candidates as-is - they should already have the correct parameter names
        return candidates
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update algorithm with evaluated points"""
        # Print input for debugging
        print(f"Tell received xs: {xs}")
        
        # Use xs as-is - they should already have the correct parameter names
        self.algorithm.tell(xs, ys)
    
    def get_result(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Get optimization result (Pareto front)"""
        pareto_xs, pareto_ys = self.algorithm.recommend()
        
        # Return parameter dictionaries without remapping - maintain original parameter names
        return pareto_xs, pareto_ys
    
    def get_hypervolume(self) -> float:
        """Get current hypervolume"""
        return self.algorithm.get_hypervolume() 