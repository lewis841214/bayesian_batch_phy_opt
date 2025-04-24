import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pymoo.core.problem
from pymoo.algorithms.moo.moead import MOEAD as PymooMOEAD
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

from src.core.algorithm import MultiObjectiveOptimizer
from src.core.parameter_space import ParameterSpace
from src.adapters.pymoo_adapter import PymooAdapter

# Reuse the problem wrapper from NSGA-II
from src.algorithms.evolutionary.nsga2 import PymooWrappedProblem

class MOEAD(MultiObjectiveOptimizer):
    """MOEA/D implementation using Pymoo backend"""
    
    def __init__(self, parameter_space: ParameterSpace, budget: int, 
                batch_size: int = 20, n_objectives: int = 2, 
                pop_size: int = 100, n_neighbors: int = 20, 
                decomposition: str = 'auto', **kwargs):
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.decomposition = decomposition
        
        # In MOEA/D, batch_size is related to the population size
        # but we use the provided batch_size if available
        if batch_size > pop_size:
            self.pop_size = batch_size
        
        super().__init__(parameter_space, budget, batch_size, n_objectives)
    
    def _setup(self):
        """Initialize Pymoo components"""
        self.adapter = PymooAdapter(self.parameter_space)
        
        # Create MOEA/D algorithm
        self.algorithm = PymooMOEAD(
            pop_size=self.pop_size,
            n_neighbors=self.n_neighbors,
            decomposition=self.decomposition,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20)
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
                n_samples=self.pop_size
            )
            
            # Set initial weights and neighbors
            self.algorithm.setup(self.problem, initial_pop)
            
            # Convert population to parameter dictionaries (get only n points)
            params_list = []
            for x in initial_pop.get("X")[:n]:
                params = self.adapter.from_framework_format(x)
                params_list.append(params)
            
            self.current_population = initial_pop
            # Store full population but return only n points
            self.current_batch_indices = list(range(min(n, len(initial_pop))))
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
        
        # Do one iteration of the algorithm to get new offspring
        # MOEA/D is different from NSGA-II in that it processes each subproblem
        # We'll implement a simplified batch-based version
        
        # Generate new offspring from the current population
        # In MOEA/D, we generate one offspring for each weight vector
        offspring = []
        for i in range(min(n, self.pop_size)):
            # Get neighbors and their indices
            N = self.algorithm.neighbors[i][:self.algorithm.n_neighbors]
            parents = self.current_population[N]
            
            # Apply variation to create new individuals
            off = self.algorithm.mating.do(
                self.problem,
                parents,
                n_offsprings=1,
                algorithm=self.algorithm
            )
            
            offspring.append(off[0])
        
        # Create population object from offspring
        from pymoo.core.population import Population
        offspring_pop = Population.create(*zip(*[(ind.X, ind.F) for ind in offspring]))
        
        # Convert offspring to parameter dictionaries
        params_list = []
        for x in offspring_pop.get("X"):
            params = self.adapter.from_framework_format(x)
            params_list.append(params)
        
        # Save current offspring
        self.current_offspring = offspring_pop
        self.current_batch_indices = list(range(len(offspring_pop)))
        
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
            
            # If we have a current population, update with new evaluations
            if self.current_population is not None:
                # For each subproblem, use MOEA/D's update strategy
                for i, idx in enumerate(self.current_batch_indices):
                    # Get current offspring
                    child = self.current_offspring[i]
                    
                    # Get the neighbors for this subproblem
                    neighbors = self.algorithm.neighbors[idx]
                    
                    # For each neighbor, check if the child is better
                    for j in neighbors:
                        # Calculate decomposed value for the child
                        val_child = self.algorithm.decomp.do(
                            child.F, self.algorithm.weights[j], self.algorithm._ideal_point)
                        
                        # Calculate decomposed value for the current solution
                        val_current = self.algorithm.decomp.do(
                            self.current_population[j].F, self.algorithm.weights[j], self.algorithm._ideal_point)
                        
                        # If child is better, replace the current solution
                        if val_child < val_current:
                            self.current_population[j] = child
                
                # Update ideal point
                self.algorithm._ideal_point = np.min(
                    np.vstack([self.algorithm._ideal_point, self.current_offspring.get("F")]), axis=0)
    
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