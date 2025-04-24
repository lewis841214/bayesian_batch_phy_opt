import numpy as np
from typing import Dict, Any, List, Tuple, Type, Optional

# Import pymoo components
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result

# Import our optimizers
from src.algorithms.bayesian.bucb import BUCB
from src.algorithms.bayesian.q_ehvi import QEHVI
from src.core.parameter_space import ParameterSpace
from src.core.algorithm import MultiObjectiveOptimizer


class PymooBayesianWrapper(Algorithm):
    """
    Wrapper for our Bayesian optimizers to make them compatible with PyMOO's API
    """
    
    def __init__(self, optimizer_class: Type[MultiObjectiveOptimizer], parameter_space: ParameterSpace, 
                 n_objectives: int, batch_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        
        self.optimizer_class = optimizer_class
        self.parameter_space = parameter_space
        self.n_objectives = n_objectives
        self.batch_size = batch_size
        self.optimizer = None
        self.kwargs = kwargs
        
        # Store evaluation history
        self.history_x = []
        self.history_y = []
        
    def _initialize(self):
        # Initialize the optimizer
        self.optimizer = self.optimizer_class(
            parameter_space=self.parameter_space,
            budget=500,  # This will be overridden by PyMOO's termination
            batch_size=self.batch_size,
            n_objectives=self.n_objectives,
            **self.kwargs
        )
        
        # Initialize the optimizer
        self.optimizer._setup()
        
    def _setup(self, problem, **kwargs):
        # Extract the number of objectives from the problem
        self.n_objectives = problem.n_obj
        
    def _ask(self):
        # Get next batch of points from our optimizer
        param_dicts = self.optimizer.ask(n=self.batch_size)
        
        # Convert to population
        X = np.empty((len(param_dicts), len(self.parameter_space.parameters)))
        
        for i, params in enumerate(param_dicts):
            for j, (name, _) in enumerate(self.parameter_space.parameters.items()):
                X[i, j] = params[name]
                
        pop = Population.new("X", X)
        return pop
    
    def _tell(self, problem, population, **kwargs):
        # Convert the population to our format and update the optimizer
        xs = []
        ys = []
        
        for i, indiv in enumerate(population):
            # Extract X values
            x_dict = {}
            for j, (name, _) in enumerate(self.parameter_space.parameters.items()):
                x_dict[name] = indiv.X[j]
            
            # Extract objective values
            y = indiv.F.tolist()
            
            xs.append(x_dict)
            ys.append(y)
            
            # Store in history
            self.history_x.append(x_dict)
            self.history_y.append(y)
        
        # Update the optimizer
        self.optimizer.tell(xs, ys)
    
    def _get_result(self):
        # Get current Pareto front
        pareto_xs, pareto_ys = self.optimizer.recommend()
        
        # Convert to the format expected by PyMOO
        result = Result()
        
        # Add history - convert to numpy arrays
        result.algorithm = self
        result.history = None  # Could implement detailed history here
        
        # Store the Pareto front and optimal solutions
        result.X = np.array([list(x.values()) for x in pareto_xs])
        result.F = np.array(pareto_ys)
        
        # Set other attributes as needed by PyMOO
        result.opt = result.X  # Optimal solutions
        
        return result 