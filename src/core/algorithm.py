from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from src.core.parameter_space import ParameterSpace

class BaseOptimizer(ABC):
    """Base class for all optimizers"""
    
    def __init__(self, parameter_space: ParameterSpace, budget: int, 
                batch_size: int = 1):
        self.parameter_space = parameter_space
        self.budget = budget
        self.batch_size = batch_size
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Initialize algorithm-specific components"""
        pass
    
    @abstractmethod
    def ask(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return batch of points to evaluate as list of parameter dictionaries"""
        pass
    
    @abstractmethod
    def tell(self, xs: List[Dict[str, Any]], ys: List):
        """Update optimizer with evaluated points"""
        pass
    
    def optimize(self, objective_function: Callable):
        """Run optimization loop"""
        remaining_budget = self.budget
        
        while remaining_budget > 0:
            # Determine batch size for this iteration
            current_batch = min(self.batch_size, remaining_budget)
            
            # Get batch of points to evaluate
            xs = self.ask(n=current_batch)
            
            # Evaluate points
            ys = [objective_function(x) for x in xs]
            
            # Update optimizer
            self.tell(xs, ys)
            
            remaining_budget -= current_batch
        
        return self.recommend()
    
    @abstractmethod
    def recommend(self):
        """Return recommended solution(s)"""
        pass

class SingleObjectiveOptimizer(BaseOptimizer):
    """Base class for single-objective optimizers"""
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[float]):
        """Update optimizer with evaluated points (single objective)"""
        pass

class MultiObjectiveOptimizer(BaseOptimizer):
    """Base class for multi-objective optimizers"""
    
    def __init__(self, parameter_space: ParameterSpace, budget: int, 
                batch_size: int = 1, n_objectives: int = 2):
        self.n_objectives = n_objectives
        super().__init__(parameter_space, budget, batch_size)
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update optimizer with evaluated points (multi-objective)"""
        pass 