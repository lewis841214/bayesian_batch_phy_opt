import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pymoo.core.problem
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.pymoo_adapter import PymooAdapter

class PymooWrappedProblem(Problem):
    """Wrapper for Pymoo problem using our adapter"""
    
    def __init__(self, parameter_space: ParameterSpace, evaluate_func, n_obj: int):
        # Get problem parameters from adapter
        self.adapter = PymooAdapter(parameter_space)
        self.pymoo_space = self.adapter.pymoo_space
        
        # Store evaluate function
        self.evaluate_func = evaluate_func
        
        # Extract dimensions
        n_var = len(parameter_space.parameters)
        xl = self.pymoo_space['bounds'][:, 0]
        xu = self.pymoo_space['bounds'][:, 1]
        
        # Initialize problem
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=True
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a single solution"""
        # If x is 1D, it's a single solution
        if x.ndim == 1:
            # Convert to parameter dictionary
            params = self.adapter.from_framework_format(x)
            
            # Evaluate
            if self.evaluate_func is not None:
                out["F"] = np.array(self.evaluate_func(params))
            else:
                out["F"] = np.zeros(self.n_obj)
        # If x is 2D, it's a batch of solutions
        else:
            # Convert each solution to parameter dictionary and evaluate
            f = []
            for xi in x:
                params = self.adapter.from_framework_format(xi)
                if self.evaluate_func is not None:
                    f.append(self.evaluate_func(params))
                else:
                    f.append(np.zeros(self.n_obj))
            
            out["F"] = np.array(f)


class NSGAII(MultiObjectiveOptimizer):
    """NSGA-II implementation using Pymoo backend"""
    
    def __init__(self, parameter_space: ParameterSpace, budget: int, 
                batch_size: int = 20, n_objectives: int = 2, 
                pop_size: int = 100, **kwargs):
        self.pop_size = pop_size
        # In NSGA-II, batch_size is actually the population size
        # but we use the provided batch_size if available
        if batch_size > pop_size:
            self.pop_size = batch_size
        
        super().__init__(parameter_space, budget, batch_size, n_objectives)
    
    def _setup(self):
        """Initialize Pymoo components"""
        self.adapter = PymooAdapter(self.parameter_space)
        
        # Create NSGA-II algorithm
        self.algorithm = PymooNSGA2(
            pop_size=self.pop_size,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        
        # Initialize data collection
        self.all_x = []
        self.all_y = []
        
        # Initialize problem (will be updated later with real evaluate function)
        self.problem = None
        
        # Track current population
        self.current_population = None
    
    def ask(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return population points for evaluation"""
        n = n or self.batch_size
        
        # Initialize if this is the first ask
        if self.problem is None:
            def dummy_evaluate(x):
                return np.zeros(self.n_objectives)
            
            self.problem = PymooWrappedProblem(
                parameter_space=self.parameter_space,
                evaluate_func=dummy_evaluate,
                n_obj=self.n_objectives
            )
            
            # Generate initial population
            initial_pop = self.algorithm.initialization.do(
                problem=self.problem, 
                n_samples=n
            )
            
            # Convert population to parameter dictionaries
            params_list = []
            for x in initial_pop.get("X"):
                params = self.adapter.from_framework_format(x)
                params_list.append(params)
            
            self.current_population = initial_pop
            return params_list
        
        # For subsequent asks, use the evolve mechanism
        # Create a dummy problem with the current evaluations
        def evaluate_func(x):
            # Look up in our database of evaluated points
            # This is a simplification; in a real implementation, we'd do proper lookup
            for i, params in enumerate(self.all_x):
                # Check if all parameters match
                match = True
                for name, value in params.items():
                    if name in x and x[name] != value:
                        match = False
                        break
                
                if match:
                    return self.all_y[i]
            
            # If not found, return zeros
            return np.zeros(self.n_objectives)
        
        # Update problem
        self.problem = PymooWrappedProblem(
            parameter_space=self.parameter_space,
            evaluate_func=evaluate_func,
            n_obj=self.n_objectives
        )
        
        # Do one iteration of the algorithm
        offspring = self.algorithm.mating.do(
            problem=self.problem,
            pop=self.current_population,
            n_offsprings=n
        )
        
        # Convert offspring to parameter dictionaries
        params_list = []
        for x in offspring.get("X"):
            params = self.adapter.from_framework_format(x)
            params_list.append(params)
        
        # Save current offspring
        self.current_offspring = offspring
        
        return params_list
    
    def tell(self, xs: List[Dict[str, Any]], ys: List[List[float]]):
        """Update algorithm with evaluated points"""
        # Store evaluations
        self.all_x.extend(xs)
        self.all_y.extend(ys)
        
        # Update offspring with evaluations
        if hasattr(self, 'current_offspring') and self.current_offspring is not None:
            # Add F values to offspring
            self.current_offspring.set("F", np.array(ys))
            
            # If we have a current population, merge and select
            if self.current_population is not None:
                # Merge and select next population
                self.current_population = self.algorithm.survival.do(
                    problem=self.problem,
                    pop=self.current_population,
                    off=self.current_offspring,
                    n_survive=self.pop_size
                )
    
    def recommend(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Return current Pareto front"""
        # If no evaluations, return empty result
        if not self.all_x or not self.all_y:
            return [], []
        
        # Convert to numpy array
        ys_array = np.array(self.all_y)
        
        # Find Pareto front using pymoo's non-dominated sorting
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting()
        fronts = nds.do(ys_array)
        pareto_indices = fronts[0]  # First front is the Pareto front
        
        # Extract Pareto solutions
        pareto_xs = [self.all_x[i] for i in pareto_indices]
        pareto_ys = [self.all_y[i] for i in pareto_indices]
        
        return pareto_xs, pareto_ys 