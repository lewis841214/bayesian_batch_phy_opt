import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.parameter_space import ParameterSpace


class TestProblem(ABC):
    """Base class for test problems"""
    
    @abstractmethod
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space for the problem"""
        pass
    
    @abstractmethod
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate the objective functions at the given parameters"""
        pass
    
    @property
    @abstractmethod
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        pass
    
    @property
    def name(self) -> str:
        """Return the name of the test problem"""
        return self.__class__.__name__
    
    def get_reference_point(self) -> List[float]:
        """Return the reference point for hypervolume calculation"""
        # Default reference point is [11.0, 11.0, ...] for each objective
        return [11.0] * self.num_objectives


class MixedParameterTestProblem(TestProblem):
    """
    Test problem with mixed parameter types
    
    This problem has:
    - 2 continuous parameters (x1, x2)
    - 1 integer parameter (x3)
    - 1 categorical parameter (x4)
    
    The objectives are:
    f1 = x1^2 + x2^2 + x3 + categorical_weight
    f2 = (x1-1)^2 + (x2-1)^2 + x3 + categorical_weight
    """
    
    def __init__(self):
        # Define categorical weights for different values
        self._categorical_weights = {
            'option_a': 0.0,  # Best option
            'option_b': 0.5,  # Medium option
            'option_c': 1.0   # Worst option
        }
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space"""
        space = ParameterSpace()
        space.add_continuous_parameter('x1', -5.0, 5.0)
        space.add_continuous_parameter('x2', -5.0, 5.0)
        space.add_integer_parameter('x3', 1, 10)
        space.add_categorical_parameter('x4', list(self._categorical_weights.keys()))
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate the objective functions"""
        x1, x2, x3 = params['x1'], params['x2'], params['x3']
        categorical_weight = self._categorical_weights[params['x4']]
        
        f1 = x1**2 + x2**2 + x3 + categorical_weight
        f2 = (x1-1)**2 + (x2-1)**2 + x3 + categorical_weight
        
        return [f1, f2]
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 2


class NonlinearTestProblem(TestProblem):
    """
    Test problem with nonlinear interactions between parameters
    
    This problem has:
    - 3 continuous parameters (x1, x2, x3)
    
    The objectives are:
    f1 = x1^2 + x2^2 + x3^2
    f2 = (x1-2)^2 + x2^2 + (x3-1)^2
    f3 = x1 + x2 + x3 + x1*x2*x3 (interaction term)
    """
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space"""
        space = ParameterSpace()
        space.add_continuous_parameter('x1', -5.0, 5.0)
        space.add_continuous_parameter('x2', -5.0, 5.0)
        space.add_continuous_parameter('x3', -5.0, 5.0)
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate the objective functions"""
        x1, x2, x3 = params['x1'], params['x2'], params['x3']
        
        f1 = x1**2 + x2**2 + x3**2
        f2 = (x1-2)**2 + x2**2 + (x3-1)**2
        f3 = x1 + x2 + x3 + x1*x2*x3  # Nonlinear interaction term
        
        return [f1, f2, f3]
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 3


class DiscreteTestProblem(TestProblem):
    """
    Test problem with mostly discrete parameters
    
    This problem has:
    - 1 continuous parameter (x1)
    - 2 integer parameters (x2, x3)
    - 2 categorical parameters (x4, x5)
    
    The objectives are:
    f1 = x1^2 + x2 + x3 + categorical_weights
    f2 = (1-x1)^2 + 10-x2 + 10-x3 + 2-categorical_weights
    """
    
    def __init__(self):
        # Define categorical weights for different values
        self._categorical_weights_x4 = {
            'low': 0.0,
            'medium': 1.0,
            'high': 2.0
        }
        
        self._categorical_weights_x5 = {
            'red': 0.0,
            'green': 0.5,
            'blue': 1.0,
            'yellow': 1.5
        }
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space"""
        space = ParameterSpace()
        space.add_continuous_parameter('x1', 0.0, 1.0)
        space.add_integer_parameter('x2', 1, 10)
        space.add_integer_parameter('x3', 1, 10)
        space.add_categorical_parameter('x4', list(self._categorical_weights_x4.keys()))
        space.add_categorical_parameter('x5', list(self._categorical_weights_x5.keys()))
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate the objective functions"""
        x1, x2, x3 = params['x1'], params['x2'], params['x3']
        
        weight_x4 = self._categorical_weights_x4[params['x4']]
        weight_x5 = self._categorical_weights_x5[params['x5']]
        total_weight = weight_x4 + weight_x5
        
        f1 = x1**2 + x2 + x3 + total_weight
        f2 = (1-x1)**2 + (10-x2) + (10-x3) + (2-total_weight)
        
        return [f1, f2]
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 2


class ConstrainedTestProblem(TestProblem):
    """
    Test problem with a constraint
    
    This problem has:
    - 2 continuous parameters (x1, x2)
    
    The objectives are:
    f1 = x1
    f2 = (1+x2)/x1
    
    with constraint: x1 > 0
    
    Note: Many optimizers don't explicitly handle constraints, so constraints
    are often implemented by making invalid solutions very bad.
    """
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space"""
        space = ParameterSpace()
        space.add_continuous_parameter('x1', 0.1, 5.0)  # Enforce x1 > 0 with bound
        space.add_continuous_parameter('x2', 0.0, 5.0)
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate the objective functions"""
        x1, x2 = params['x1'], params['x2']
        
        f1 = x1
        f2 = (1 + x2) / x1
        
        return [f1, f2]
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 2


# Define available test problems
TEST_PROBLEMS = {
    'mixed': MixedParameterTestProblem(),
    'nonlinear': NonlinearTestProblem(),
    'discrete': DiscreteTestProblem(),
    'constrained': ConstrainedTestProblem()
}


def get_test_problem(name: str) -> TestProblem:
    """Get a test problem by name"""
    if name not in TEST_PROBLEMS:
        raise ValueError(f"Unknown test problem: {name}. Available problems: {list(TEST_PROBLEMS.keys())}")
    return TEST_PROBLEMS[name]


def list_test_problems() -> List[str]:
    """Return a list of available test problems"""
    return list(TEST_PROBLEMS.keys()) 