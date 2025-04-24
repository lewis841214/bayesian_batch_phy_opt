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



class LargeMixedParameterTestProblem(TestProblem):
    """
    Test problem with mixed parameter types
    
    This problem has:
    - 2 continuous parameters (x1, x2)
    - 1 integer parameter (x3)
    - 11 categorical parameter (x4)
    
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


class CategoryMatrixTestProblem(TestProblem):
    """
    Test problem with mostly categorical parameters
    
    This problem has:
    - 15 categorical parameters (x1 through x15), each with 4 options
    
    Each option maps to a 4x15 weight matrix, and the evaluation integrates
    these matrices to produce two objective values.
    """
    
    def __init__(self):
        # Number of parameters and options
        self.num_params = 15
        self.num_options = 4
        self.matrix_rows = 4
        
        # Generate weight matrices for each option of each parameter
        # Each option has a 4x15 matrix of weights
        self.option_weights = {}
        
        # Seed for reproducibility
        np.random.seed(42)
        
        # For each parameter
        for param_idx in range(1, self.num_params + 1):
            param_name = f'x{param_idx}'
            self.option_weights[param_name] = {}
            
            # For each option of this parameter
            for option_idx in range(self.num_options):
                option_name = f'option_{chr(97 + option_idx)}'  # a, b, c, d
                
                # Create a 4x15 weight matrix for this option
                # We'll make some options better than others, but with different trade-offs
                weights = np.zeros((self.matrix_rows, self.num_params))
                
                # Make different options good for different objectives
                # This creates a clearer trade-off between objectives
                if option_idx == 0:
                    # Good for objective 1, bad for objective 2
                    weights[0:2, :] = np.random.uniform(0.1, 0.3, size=(2, self.num_params))  # Low values (good for obj1)
                    weights[2:4, :] = np.random.uniform(0.7, 0.9, size=(2, self.num_params))  # High values (bad for obj2)
                elif option_idx == 1:
                    # Bad for objective 1, good for objective 2
                    weights[0:2, :] = np.random.uniform(0.7, 0.9, size=(2, self.num_params))  # High values (bad for obj1)
                    weights[2:4, :] = np.random.uniform(0.1, 0.3, size=(2, self.num_params))  # Low values (good for obj2)
                elif option_idx == 2:
                    # Balanced but not optimal for either
                    weights[:, :] = np.random.uniform(0.4, 0.6, size=(self.matrix_rows, self.num_params))
                else:
                    # Random - could be good or bad
                    weights[:, :] = np.random.uniform(0.1, 0.9, size=(self.matrix_rows, self.num_params))
                
                self.option_weights[param_name][option_name] = weights
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space with 15 categorical parameters"""
        space = ParameterSpace()
        
        # Add 15 categorical parameters
        options = [f'option_{chr(97 + i)}' for i in range(self.num_options)]  # a, b, c, d
        
        for i in range(1, self.num_params + 1):
            space.add_categorical_parameter(f'x{i}', options)
            
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """
        Evaluate the objective functions
        
        1. Get the weight matrix for each selected option
        2. Sum these matrices to get a combined weight matrix
        3. Apply different transformations to get two objective values
        """
        # Initialize result matrices
        combined_matrix = np.zeros((self.matrix_rows, self.num_params))
        
        # For each parameter, add its option's weight matrix to the combined matrix
        for param_idx in range(1, self.num_params + 1):
            param_name = f'x{param_idx}'
            selected_option = params[param_name]
            option_matrix = self.option_weights[param_name][selected_option]
            
            # Add this option's matrix to the combined matrix
            combined_matrix += option_matrix
        
        # Calculate two explicitly conflicting objective values from the combined matrix
        # Objective 1: Sum of first half of rows (minimize)
        f1 = np.sum(combined_matrix[0:2, :])
        
        # Objective 2: Sum of second half of rows (minimize)
        f2 = np.sum(combined_matrix[2:4, :])
        
        return [f1, f2]
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 2


class ComplexCategoryEmbeddingProblem(TestProblem):
    """
    Complex test problem with categorical parameters using embedding vectors
    
    This problem has:
    - 15 categorical parameters (x1 through x15), each with 4 options
    
    Each option maps to an embedding vector. When a parameter value is selected,
    its embedding vector is extracted. All selected embeddings form a matrix
    which is processed through non-linear transformations to produce two objectives.
    """
    
    def __init__(self, n_embed=12):
        # Number of parameters and options
        self.num_params = 8
        self.num_options = 3
        self.n_embed = n_embed
        
        # Seed for reproducibility
        np.random.seed(42)
        
        # For each parameter (i), create a matrix of shape [num_options x n_embed]
        # mp[i][j] gives the embedding vector for option j of parameter i
        self.param_embeddings = {}
        
        for param_idx in range(1, self.num_params + 1):
            param_name = f'x{param_idx}'
            # Create embeddings for each option of this parameter
            option_embeddings = {}
            
            for option_idx in range(self.num_options):
                option_name = f'option_{chr(97 + option_idx)}'  # a, b, c, d
                # Create an embedding vector for this option with different patterns
                if option_idx == 0:
                    # Option A: Strong positive values in first half, negative in second half
                    embedding = np.concatenate([
                        np.random.normal(1.0, 0.3, size=n_embed//2),
                        np.random.normal(-1.0, 0.3, size=n_embed//2)
                    ])
                elif option_idx == 1:
                    # Option B: Strong negative values in first half, positive in second half
                    embedding = np.concatenate([
                        np.random.normal(-1.0, 0.3, size=n_embed//2),
                        np.random.normal(1.0, 0.3, size=n_embed//2)
                    ])
                elif option_idx == 2:
                    # Option C: Alternating positive and negative
                    embedding = np.zeros(n_embed)
                    embedding[::2] = np.random.normal(1.0, 0.3, size=n_embed//2)
                    embedding[1::2] = np.random.normal(-1.0, 0.3, size=n_embed//2)
                else:
                    # Option D: Random values
                    embedding = np.random.normal(0, 1.0, size=n_embed)
                
                option_embeddings[option_name] = embedding
                
            self.param_embeddings[param_name] = option_embeddings
        
        # Create weight matrices for non-linear transformations
        # First transformation: [15 x n_embed] -> [12 x n_embed]
        self.W1 = np.random.normal(0, 0.5, size=(12, self.num_params))
        # Second transformation: [12 x n_embed] -> [8 x n_embed]
        self.W2 = np.random.normal(0, 0.5, size=(8, 12))
        # Final transformation: [8 x n_embed] -> [2]
        self.W_final = np.random.normal(0, 0.5, size=(2, 8, n_embed))
        
        # Create biases
        self.b1 = np.random.normal(0, 0.1, size=(12, 1))
        self.b2 = np.random.normal(0, 0.1, size=(8, 1))
        self.b_final = np.random.normal(0, 0.1, size=2)
    
    def get_parameter_space(self) -> ParameterSpace:
        """Return the parameter space with 15 categorical parameters"""
        space = ParameterSpace()
        
        # Add 15 categorical parameters
        options = [f'option_{chr(97 + i)}' for i in range(self.num_options)]  # a, b, c, d
        
        for i in range(1, self.num_params + 1):
            space.add_categorical_parameter(f'x{i}', options)
            
        return space
    
    def evaluate(self, params: Dict[str, Any]) -> List[float]:
        """
        Evaluate the objective functions using embeddings and non-linear transformations
        
        1. Get embedding vectors for each selected option
        2. Form a matrix from these embeddings
        3. Apply non-linear transformations
        4. Output two objective values
        """
        # Get embedding for each parameter's selected option
        # This forms a matrix of shape [num_params x n_embed]
        embedding_matrix = np.zeros((self.num_params, self.n_embed))
        
        for param_idx in range(1, self.num_params + 1):
            param_name = f'x{param_idx}'
            selected_option = params[param_name]
            embedding = self.param_embeddings[param_name][selected_option]
            embedding_matrix[param_idx-1] = embedding
        
        # Apply non-linear transformations
        # First layer: [12 x 15] x [15 x n_embed] = [12 x n_embed]
        hidden1 = np.dot(self.W1, embedding_matrix)
        # Apply Leaky ReLU activation
        hidden1 = np.maximum(0.1 * hidden1, hidden1)
        
        # Second layer: [8 x 12] x [12 x n_embed] = [8 x n_embed]
        hidden2 = np.dot(self.W2, hidden1)
        # Apply Tanh activation
        hidden2 = np.tanh(hidden2)
        
        # Final layer - produce 2 objective values
        # For each objective, apply a different transformation
        objectives = np.zeros(2)
        
        # Split the embeddings in half for different objective calculations
        half_embed = self.n_embed // 2
        
        # Objective 1: Focus on first half of embedding dimensions
        weighted_sums1 = np.sum(self.W_final[0, :, :half_embed] * hidden2[:, :half_embed], axis=(0, 1))
        
        # Objective 2: Focus on second half of embedding dimensions
        weighted_sums2 = np.sum(self.W_final[1, :, half_embed:] * hidden2[:, half_embed:], axis=(0, 1))
        
        # Conflict between objectives through highly non-linear transformations
        objectives[0] = 100 * np.exp(np.sin(weighted_sums1) + self.b_final[0]) 
        objectives[1] = 100 * np.exp(np.cos(weighted_sums2) + self.b_final[1])
        
        # # Additional non-linear interaction between objectives
        # # This creates a more interesting Pareto front shape
        # objectives[0] += 10 * np.sin(weighted_sums2 * 1.5)
        # objectives[1] += 10 * np.cos(weighted_sums1 * 1.5)
        
        # # Make the objectives even more conflicting
        # shared_term = np.sum(embedding_matrix[:, half_embed//2:half_embed])
        # objectives[0] += 50 * np.sin(shared_term)
        # objectives[1] += 50 * np.cos(shared_term)
        
        # # Scale objectives to create more diversity
        # objectives[0] = 200 + objectives[0] * 50
        # objectives[1] = 200 + objectives[1] * 50
        
        return objectives.tolist()
    
    @property
    def num_objectives(self) -> int:
        """Return the number of objectives"""
        return 2


# Define available test problems
TEST_PROBLEMS = {
    'mixed': MixedParameterTestProblem(),
    'nonlinear': NonlinearTestProblem(),
    'discrete': DiscreteTestProblem(),
    'constrained': ConstrainedTestProblem(),
    'large_mixed': LargeMixedParameterTestProblem(),
    'category_matrix': CategoryMatrixTestProblem(),
    'complex_categorical': ComplexCategoryEmbeddingProblem()
}


def get_test_problem(name: str) -> TestProblem:
    """Get a test problem by name"""
    if name not in TEST_PROBLEMS:
        raise ValueError(f"Unknown test problem: {name}. Available problems: {list(TEST_PROBLEMS.keys())}")
    return TEST_PROBLEMS[name]


def list_test_problems() -> List[str]:
    """Return a list of available test problems"""
    return list(TEST_PROBLEMS.keys()) 