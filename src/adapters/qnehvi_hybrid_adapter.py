from typing import Dict, List, Any, Optional, Tuple, Union

from src.adapters.algorithm_adapters import BayesianOptAdapter
from src.algorithms.bayesian.neural_qnehvi import NNQNEHVI

class QNEHVIHybridAdapter(BayesianOptAdapter):
    """
    Adapter for Neural Network enhanced qNEHVI Bayesian optimization
    
    This adapter handles the integration of neural network surrogate models
    with the standard qNEHVI algorithm for multi-objective optimization.
    """
    
    def __init__(self, surrogate_model="nn", nn_config=None):
        """
        Initialize adapter with neural network configuration
        
        Args:
            surrogate_model: Type of surrogate model to use ('nn' or 'xgboost')
            nn_config: Optional configuration for neural network
        """
        self.surrogate_model = surrogate_model
        self.nn_config = nn_config or {
            'hidden_layers': [10], # [10, 10], two hidden layers, each with 10 neurons
            'learning_rate': 0.1,
            'epochs': 3000,
            'batch_size': 100,
            'regularization': 1e-5
        }
        
        # Initialize with NNQNEHVI algorithm class
        super().__init__(NNQNEHVI)
    
    def setup(self, problem, budget, batch_size):
        """
        Setup Neural Network enhanced Bayesian optimizer
        
        Args:
            problem: Optimization problem
            budget: Total evaluation budget
            batch_size: Number of points to evaluate in parallel
            
        Returns:
            Configured optimizer instance
        """
        parameter_space = problem.get_parameter_space()
        ref_point = problem.get_reference_point()
        
        # Initialize algorithm with neural network configuration
        self.algorithm = NNQNEHVI(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128,
            
            # Neural network specific parameters
            nn_layers=self.nn_config['hidden_layers'],
            nn_learning_rate=self.nn_config['learning_rate'],
            nn_epochs=self.nn_config['epochs'],
            nn_batch_size=self.nn_config['batch_size'],
            nn_regularization=self.nn_config['regularization'],
        )
        
        return self
    
    def ask(self, output_dir=None):
        """Get next batch of candidates with support for model prediction plots"""
        if output_dir is not None:
            return self.algorithm.ask(output_dir=output_dir)
        return self.algorithm.ask()
    
    def get_model_metrics(self):
        """Return neural network training metrics"""
        if hasattr(self.algorithm, 'get_model_metrics'):
            return self.algorithm.get_model_metrics()
        return {} 