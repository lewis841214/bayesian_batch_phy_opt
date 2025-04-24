import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type

from src.algorithms.bayesian.qnehvi import QNEHVI
from src.algorithms.bayesian.qehvi import QEHVI
from src.algorithms.bayesian.qparego import QNParEGO

# Import evolutionary algorithm classes from pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2 as PyMOO_NSGA2
from pymoo.algorithms.moo.moead import MOEAD as PyMOO_MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3 as PyMOO_NSGA3


class AlgorithmAdapter:
    """Base adapter class for wrapping different optimization algorithms"""
    
    def setup(self, problem, budget, batch_size):
        """Setup the algorithm with problem parameters"""
        pass
    
    def ask(self):
        """Get next batch of candidate solutions"""
        pass
    
    def tell(self, x, y):
        """Update with evaluated solutions"""
        pass
    
    def get_result(self):
        """Return current result (Pareto front)"""
        pass
    
    def get_hypervolume(self):
        """Calculate current hypervolume if supported"""
        pass


class BayesianOptAdapter(AlgorithmAdapter):
    """Adapter for Bayesian optimization algorithms"""
    
    def __init__(self, algorithm_class):
        """Initialize with algorithm class (QNEHVI, QEHVI, QNParEGO)"""
        self.algorithm_class = algorithm_class
        self.algorithm = None
    
    def setup(self, problem, budget, batch_size):
        """Setup Bayesian optimizer"""
        parameter_space = problem.get_parameter_space()
        ref_point = problem.get_reference_point()
        
        self.algorithm = self.algorithm_class(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128
        )
        return self
    
    def ask(self):
        """Get next batch of candidates"""
        return self.algorithm.ask()
    
    def tell(self, x, y):
        """Update with evaluated solutions"""
        self.algorithm.tell(x, y)
    
    def get_result(self):
        """Return current result"""
        return self.algorithm.recommend()
    
    def get_hypervolume(self):
        """Calculate hypervolume"""
        return self.algorithm.get_hypervolume()


class EvolutionaryAdapter(AlgorithmAdapter):
    """Adapter for evolutionary algorithms from pymoo"""
    
    def __init__(self, algorithm_class, algorithm_params=None):
        """Initialize with algorithm class and optional parameters"""
        self.algorithm_class = algorithm_class
        self.algorithm_params = algorithm_params or {}
        self.algorithm = None
        self.pymoo_problem = None
        self.current_gen = 0
        self.max_gen = 0
        self.evaluated_x = []
        self.evaluated_y = []
        self.progress = {}
        self.is_initialized = False
    
    def setup(self, problem, budget, batch_size):
        """Setup evolutionary optimizer using pymoo interfaces"""
        from pymoo.core.problem import Problem
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.termination.max_gen import MaximumGenerationTermination
        
        # Create pymoo problem wrapper
        class PymooProblem(Problem):
            def __init__(self, test_problem):
                self.test_problem = test_problem
                param_space = test_problem.get_parameter_space()
                
                # Extract bounds for each parameter
                xl, xu = [], []
                param_types = []
                self.param_names = list(param_space.parameters.keys())
                
                for name, param in param_space.parameters.items():
                    if param['type'] in ['continuous', 'integer']:
                        xl.append(param['bounds'][0])
                        xu.append(param['bounds'][1])
                        param_types.append(param['type'])
                    elif param['type'] == 'categorical':
                        # Map categorical to integers
                        xl.append(0)
                        xu.append(len(param['categories']) - 1)
                        param_types.append('categorical')
                        
                super().__init__(
                    n_var=len(self.param_names),
                    n_obj=test_problem.num_objectives,
                    n_constr=0,
                    xl=np.array(xl),
                    xu=np.array(xu)
                )
                self.param_types = param_types
                self.categorical_mappings = {}
                
                # Setup categorical mappings
                for i, name in enumerate(self.param_names):
                    param = param_space.parameters[name]
                    if param['type'] == 'categorical':
                        self.categorical_mappings[i] = param['categories']
            
            def _evaluate(self, x, out, *args, **kwargs):
                # Convert to problem's parameter format
                n_points = x.shape[0]
                f_values = []
                
                for i in range(n_points):
                    params = {}
                    for j, name in enumerate(self.param_names):
                        if self.param_types[j] == 'continuous':
                            params[name] = float(x[i, j])
                        elif self.param_types[j] == 'integer':
                            params[name] = int(round(float(x[i, j])))
                        elif self.param_types[j] == 'categorical':
                            cat_idx = int(round(float(x[i, j])))
                            params[name] = self.categorical_mappings[j][cat_idx]
                    
                    # Evaluate
                    f = self.test_problem.evaluate(params)
                    f_values.append(f)
                
                out["F"] = np.array(f_values) * -1  # Negate for maximization
        
        # Setup problem wrapper
        self.test_problem = problem
        self.pymoo_problem = PymooProblem(problem)
        self.ref_point = np.array(problem.get_reference_point()) * -1  # Negate for maximization
        
        # Calculate generations based on budget and batch size
        self.max_gen = budget // batch_size
        self.termination = MaximumGenerationTermination(n_max_gen=self.max_gen)
        
        # Set up algorithm specific parameters
        if self.algorithm_class == PyMOO_NSGA3:
            ref_dirs = get_reference_directions("das-dennis", problem.num_objectives, n_partitions=12)
            self.algorithm = self.algorithm_class(
                pop_size=batch_size,
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                ref_dirs=ref_dirs,
                **self.algorithm_params
            )
        elif self.algorithm_class == PyMOO_MOEAD:
            ref_dirs = get_reference_directions("das-dennis", problem.num_objectives, n_partitions=12)
            self.algorithm = self.algorithm_class(
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                ref_dirs=ref_dirs,
                **self.algorithm_params
            )
        else:  # Default (NSGA2)
            self.algorithm = self.algorithm_class(
                pop_size=batch_size,
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                **self.algorithm_params
            )
        
        self.current_gen = 0
        self.batch_size = batch_size
        
        # Initialize algorithm with problem
        self.algorithm.setup(self.pymoo_problem)
        
        # Create initial population manually
        from pymoo.core.population import Population
        pop = self.algorithm.initialization.do(self.pymoo_problem, self.algorithm.pop_size)
        
        # Evaluate initial population
        self.pymoo_problem.evaluate(pop, out={"F": None})
        
        # Set the algorithm's population
        self.algorithm.pop = pop
        
        # Store population data
        self._update_evaluations()
        
        self.is_initialized = True
        return self
    
    def _update_evaluations(self):
        """Update the evaluation history from current population"""
        # Extract X and F from population
        if self.algorithm.pop is None:
            return
            
        X = self.algorithm.pop.get("X")
        F = self.algorithm.pop.get("F") * -1  # Unnegate
        
        if X is None or F is None:
            return
        
        # Convert to parameter dictionaries
        self.evaluated_x = []
        self.evaluated_y = []
        
        for i in range(len(X)):
            x_params = {}
            for j, name in enumerate(self.pymoo_problem.param_names):
                if self.pymoo_problem.param_types[j] == 'continuous':
                    x_params[name] = float(X[i, j])
                elif self.pymoo_problem.param_types[j] == 'integer':
                    x_params[name] = int(round(float(X[i, j])))
                elif self.pymoo_problem.param_types[j] == 'categorical':
                    cat_idx = int(round(float(X[i, j])))
                    x_params[name] = self.pymoo_problem.categorical_mappings[j][cat_idx]
            
            self.evaluated_x.append(x_params)
            self.evaluated_y.append(F[i].tolist())
    
    def ask(self):
        """Get current population as candidates"""
        if not self.is_initialized or not self.algorithm or not self.algorithm.pop:
            return []
        
        X = self.algorithm.pop.get("X")
        if X is None:
            return []
        
        # Convert to parameter dictionaries
        candidates = []
        for i in range(len(X)):
            x_params = {}
            for j, name in enumerate(self.pymoo_problem.param_names):
                if self.pymoo_problem.param_types[j] == 'continuous':
                    x_params[name] = float(X[i, j])
                elif self.pymoo_problem.param_types[j] == 'integer':
                    x_params[name] = int(round(float(X[i, j])))
                elif self.pymoo_problem.param_types[j] == 'categorical':
                    cat_idx = int(round(float(X[i, j])))
                    x_params[name] = self.pymoo_problem.categorical_mappings[j][cat_idx]
            
            candidates.append(x_params)
        
        return candidates
    
    def tell(self, x, y):
        """Update with evaluated solutions and run the next generation"""
        # Check if properly initialized before continuing
        if not self.is_initialized:
            raise RuntimeError("Algorithm not properly initialized. Call setup() first.")
            
        # Increment generation counter
        self.current_gen += 1
        
        if self.current_gen >= self.max_gen:
            return
           
        # Set current population's F values directly (already evaluated externally)
        F = -1 * np.array(y)  # Negate for maximization
        for i in range(len(self.algorithm.pop)):
            self.algorithm.pop[i].set("F", F[i])
        
        # Advance to the next generation without using deepcopy
        try:
            # This performs mating and generates offspring
            self.algorithm.mating.do(self.algorithm.problem, self.algorithm.pop, self.algorithm.n_offsprings, algorithm=self.algorithm)
            
            # Store newly evaluated solutions
            self._update_evaluations()
            
            # Create new population for next generation
            self.algorithm.next()
            
        except Exception as e:
            print(f"Error in evolutionary algorithm: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_result(self):
        """Return current Pareto front"""
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        
        # Find non-dominated solutions
        if len(self.evaluated_y) == 0:
            return [], []
        
        F = -1 * np.array(self.evaluated_y)  # Negate for sorting
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        
        # Extract Pareto front
        pareto_x = [self.evaluated_x[i] for i in nds]
        pareto_y = [self.evaluated_y[i] for i in nds]
        
        return pareto_x, pareto_y
    
    def get_hypervolume(self):
        """Calculate hypervolume of current Pareto front"""
        from pymoo.indicators.hv import Hypervolume
        
        if len(self.evaluated_y) == 0:
            return 0.0
            
        # Get Pareto front
        pareto_x, pareto_y = self.get_result()
        
        if len(pareto_y) == 0:
            return 0.0
            
        F = -1 * np.array(pareto_y)  # Convert to maximization
        hv = Hypervolume(ref_point=self.ref_point)
        return hv.do(F)


def get_algorithm_adapter(algorithm_name):
    """Get appropriate adapter for algorithm name"""
    # Bayesian optimization algorithms
    if algorithm_name.lower() == 'qnehvi':
        return BayesianOptAdapter(QNEHVI)
    elif algorithm_name.lower() == 'qehvi':
        return BayesianOptAdapter(QEHVI)
    elif algorithm_name.lower() == 'qparego':
        return BayesianOptAdapter(QNParEGO)
    
    # Evolutionary algorithms
    elif algorithm_name.lower() == 'nsga2':
        return EvolutionaryAdapter(PyMOO_NSGA2)
    elif algorithm_name.lower() == 'moead':
        return EvolutionaryAdapter(PyMOO_MOEAD)
    elif algorithm_name.lower() == 'nsga3':
        return EvolutionaryAdapter(PyMOO_NSGA3)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}") 