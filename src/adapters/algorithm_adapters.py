import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type

from src.algorithms.bayesian.qnehvi import QNEHVI
# from src.algorithms.bayesian.qehvi import QEHVI
# from src.algorithms.bayesian.qparego import QNParEGO

# Import random search algorithm
from src.algorithms.random.random_search import RandomSearch

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
    
    def ask(self, output_dir=None):
        """Get next batch of candidates"""
        if output_dir is not None and hasattr(self.algorithm, "ask") and "output_dir" in self.algorithm.ask.__code__.co_varnames:
            return self.algorithm.ask(output_dir=output_dir)
        return self.algorithm.ask()
    
    def tell(self, x, y, hidden_maps=None):
        """Update with evaluated solutions"""
        if hidden_maps is not None:
            self.algorithm.tell(x, y, hidden_maps)
        else:
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
        self.pymoo_problem = None
        self.current_gen = 0
        self.max_gen = 0
        self.evaluated_x = []
        self.evaluated_y = []
        # Keep a separate history of all unique evaluations
        self.all_evaluated_x = []
        self.all_evaluated_y = []
        # Dictionary to track unique evaluations (for faster lookup)
        self.evaluation_cache = {}
    
    def setup(self, problem, budget, batch_size):
        """Setup evolutionary optimizer using pymoo interfaces"""
        from pymoo.core.problem import Problem
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.util.ref_dirs import get_reference_directions
        
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
                # Handle different input types
                from pymoo.core.individual import Individual
                
                # Check if input is a list/array of Individual objects
                if isinstance(x[0], Individual):
                    # Extract X values from individuals
                    X = np.array([ind.X for ind in x])
                else:
                    X = x
                
                # Ensure X is 2D even if only one solution
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                
                n_points = X.shape[0]
                f_values = []
                
                for i in range(n_points):
                    params = {}
                    for j, name in enumerate(self.param_names):
                        if self.param_types[j] == 'continuous':
                            params[name] = float(X[i, j])
                        elif self.param_types[j] == 'integer':
                            params[name] = int(round(float(X[i, j])))
                        elif self.param_types[j] == 'categorical':
                            cat_idx = int(round(float(X[i, j])))
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
        self.batch_size = batch_size
        self.current_gen = 0
        
        # Reset tracking variables
        self.evaluated_x = []
        self.evaluated_y = []
        self.all_evaluated_x = []
        self.all_evaluated_y = []
        self.evaluation_cache = {}
        
        # Run first generation to initialize
        self._run_generation()
        
        return self
    
    def _create_parameter_key(self, params):
        """Create a unique key for parameter dictionary for caching"""
        # Sort keys to ensure consistent ordering
        keys = sorted(params.keys())
        return tuple((k, params[k]) for k in keys)
    
    def _is_duplicate(self, params):
        """Check if parameters have been evaluated before"""
        key = self._create_parameter_key(params)
        return key in self.evaluation_cache
    
    def _add_evaluation(self, params, f_value):
        """Add evaluation to history if it's unique"""
        key = self._create_parameter_key(params)
        
        # Only add if it's not a duplicate
        if key not in self.evaluation_cache:
            self.evaluation_cache[key] = len(self.all_evaluated_y)  # Store index for faster lookup
            self.all_evaluated_x.append(params)
            self.all_evaluated_y.append(f_value)
            return True
        return False

    def _create_algorithm(self):
        """Create a fresh instance of the algorithm"""
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.util.ref_dirs import get_reference_directions
        
        if self.algorithm_class == PyMOO_NSGA3:
            ref_dirs = get_reference_directions("das-dennis", self.test_problem.num_objectives, n_partitions=12)
            return self.algorithm_class(
                pop_size=self.batch_size,
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                ref_dirs=ref_dirs,
                **self.algorithm_params
            )
        elif self.algorithm_class == PyMOO_MOEAD:
            ref_dirs = get_reference_directions("das-dennis", self.test_problem.num_objectives, n_partitions=12)
            return self.algorithm_class(
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                ref_dirs=ref_dirs,
                **self.algorithm_params
            )
        else:  # Default (NSGA2)
            return self.algorithm_class(
                pop_size=self.batch_size,
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                **self.algorithm_params
            )
    
    def _run_generation(self):
        """Run a single generation of the evolutionary algorithm"""
        from pymoo.optimize import minimize
        
        algorithm = self._create_algorithm()
        res = minimize(
            self.pymoo_problem,
            algorithm,
            termination=('n_gen', self.current_gen + 1),
            seed=42,
            verbose=False
        )
        
        # Extract and store results
        X = res.pop.get("X")
        F = res.pop.get("F") * -1  # Unnegate
        
        # Update current population
        self.evaluated_x = []
        self.evaluated_y = []
        
        # Convert all evaluations to parameter dictionaries
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
            
            # Add to current population
            self.evaluated_x.append(x_params)
            self.evaluated_y.append(F[i].tolist())
            
            # Add to all-time history if unique
            self._add_evaluation(x_params, F[i].tolist())
            
        return res
    
    def ask(self):
        """Get current population as candidates"""
        if not self.evaluated_x:
            return []
        
        return self.evaluated_x
    
    def tell(self, x, y):
        """Update with evaluated solutions and run the next generation"""
        # Add new evaluations to history if they're unique
        for i in range(len(x)):
            self._add_evaluation(x[i], y[i])
            
        # Increment generation counter
        self.current_gen += 1
        
        if self.current_gen >= self.max_gen:
            return
            
        # Run the next generation
        try:
            self._run_generation()
        except Exception as e:
            print(f"Error in evolutionary algorithm: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_result(self):
        """Return current Pareto front"""
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        
        # Find non-dominated solutions from all unique evaluations
        if len(self.all_evaluated_y) == 0:
            return [], []
        
        F = -1 * np.array(self.all_evaluated_y)  # Negate for sorting
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        
        # Extract Pareto front
        pareto_x = [self.all_evaluated_x[i] for i in nds]
        pareto_y = [self.all_evaluated_y[i] for i in nds]
        
        return pareto_x, pareto_y
    
    def get_hypervolume(self):
        """Calculate hypervolume of current Pareto front"""
        from pymoo.indicators.hv import Hypervolume
        
        if len(self.all_evaluated_y) == 0:
            return 0.0
            
        # Get Pareto front
        pareto_x, pareto_y = self.get_result()
        
        if len(pareto_y) == 0:
            return 0.0
            
        F = -1 * np.array(pareto_y)  # Convert to maximization
        hv = Hypervolume(ref_point=self.ref_point)
        return hv.do(F)


class RandomSearchAdapter(AlgorithmAdapter):
    """Adapter for the random search algorithm"""
    
    def __init__(self):
        """Initialize adapter"""
        self.algorithm = None
    
    def setup(self, problem, budget, batch_size):
        """Setup random search optimizer"""
        parameter_space = problem.get_parameter_space()
        ref_point = problem.get_reference_point()
        
        self.algorithm = RandomSearch(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point
        )
        return self
    
    def ask(self):
        """Get next batch of points to evaluate"""
        return self.algorithm.ask()
    
    def tell(self, x, y):
        """Update with evaluated solutions"""
        self.algorithm.tell(x, y)
    
    def get_result(self):
        """Return current result (Pareto front)"""
        return self.algorithm.recommend()
    
    def get_hypervolume(self):
        """Calculate hypervolume"""
        return self.algorithm.get_hypervolume()


def get_algorithm_adapter(algorithm_name: str) -> AlgorithmAdapter:
    """Get algorithm adapter based on name"""
    algorithm_name = algorithm_name.lower()
    
    # Bayesian optimization algorithms
    if algorithm_name == 'qnehvi':
        return BayesianOptAdapter(QNEHVI)
    # elif algorithm_name == 'qehvi':
    #     return BayesianOptAdapter(QEHVI)
    # elif algorithm_name == 'qparego':
    #     return BayesianOptAdapter(QNParEGO)
    
    # Evolutionary algorithms
    elif algorithm_name == 'nsga2':
        return EvolutionaryAdapter(PyMOO_NSGA2)
    elif algorithm_name == 'moead':
        return EvolutionaryAdapter(PyMOO_MOEAD)
    elif algorithm_name == 'nsga3':
        return EvolutionaryAdapter(PyMOO_NSGA3)
    
    # Special case for hybrid methods
    elif algorithm_name == 'nn-qnehvi':
        from src.adapters.qnehvi_hybrid_adapter import QNEHVIHybridAdapter
        return QNEHVIHybridAdapter(surrogate_model="nn")
    elif algorithm_name == 'xgb-qnehvi':
        from src.adapters.qnehvi_hybrid_adapter import QNEHVIHybridAdapter
        return QNEHVIHybridAdapter(surrogate_model="xgboost")
    elif algorithm_name == 'randomsearch':
        return RandomSearchAdapter()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}") 