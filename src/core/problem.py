from typing import List, Dict, Any, Callable, Optional
from src.core.parameter_space import ParameterSpace

class MultiObjectiveProblem:
    """Definition of a multi-objective optimization problem"""
    
    def __init__(self, parameter_space: ParameterSpace, 
                objective_functions: List[Callable[[Dict[str, Any]], float]], 
                reference_point: Optional[List[float]] = None):
        self.parameter_space = parameter_space
        self.objective_functions = objective_functions
        self.reference_point = reference_point
    
    def evaluate(self, x: Dict[str, Any]) -> List[float]:
        """Evaluate a single point on all objectives"""
        return [f(x) for f in self.objective_functions]
    
    def evaluate_batch(self, xs: List[Dict[str, Any]]) -> List[List[float]]:
        """Evaluate a batch of points on all objectives"""
        return [self.evaluate(x) for x in xs] 