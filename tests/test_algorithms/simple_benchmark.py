#!/usr/bin/env python3
import argparse
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Type, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test problems
from tests.test_algorithms.test_problems import TestProblem, get_test_problem, list_test_problems

# Import pymoo components
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

# Import our Bayesian optimization algorithms
from src.algorithms.bayesian.bucb import BUCB
from src.algorithms.bayesian.q_ehvi import QEHVI
from src.adapters.pymoo_wrapper import PymooBayesianWrapper

# Dictionary of available optimization algorithms
OPTIMIZERS = {
    'nsga2': NSGA2,
    'moead': MOEAD,
    'nsga3': NSGA3,
    'bucb': None,  # Will be set up in run_optimizer_on_problem
    'qehvi': None,  # Will be set up in run_optimizer_on_problem
}

class PymooWrappedProblem(Problem):
    """Wrapper for test problems to use with pymoo"""
    
    def __init__(self, test_problem: TestProblem):
        self.test_problem = test_problem
        self.param_space = test_problem.get_parameter_space()
        
        # Extract dimensions
        n_var = len(self.param_space.parameters)
        
        # Calculate bounds
        xl = []
        xu = []
        
        # Track variables that need to be rounded to integers
        self.integer_vars = []
        self.categorical_vars = []
        self.categorical_maps = {}
        
        i = 0
        for name, param_config in self.param_space.parameters.items():
            if param_config['type'] == 'continuous':
                xl.append(param_config['bounds'][0])
                xu.append(param_config['bounds'][1])
            elif param_config['type'] == 'integer':
                xl.append(param_config['bounds'][0])
                xu.append(param_config['bounds'][1])
                self.integer_vars.append(i)
            elif param_config['type'] == 'categorical':
                categories = param_config['categories']
                xl.append(0)
                xu.append(len(categories) - 1)
                self.categorical_vars.append(i)
                self.categorical_maps[i] = {
                    'name': name,
                    'map': {j: cat for j, cat in enumerate(categories)}
                }
            i += 1
        
        # Initialize problem
        super().__init__(
            n_var=n_var,
            n_obj=test_problem.num_objectives,
            n_constr=0,
            xl=np.array(xl),
            xu=np.array(xu),
            elementwise_evaluation=True
        )
    
    def _convert_to_param_dict(self, x):
        """Convert numpy array to parameter dictionary"""
        result = {}
        i = 0
        
        for name, param_config in self.param_space.parameters.items():
            if i in self.integer_vars:
                # Round to nearest integer
                value = x[i]
                if hasattr(value, "__len__"):
                    result[name] = int(np.round(value[0]))
                else:
                    result[name] = int(np.round(value))
            elif i in self.categorical_vars:
                # Map integer to category
                value = x[i]
                if hasattr(value, "__len__"):
                    cat_idx = int(np.round(value[0]))
                else:
                    cat_idx = int(np.round(value))
                
                # Ensure category index is within bounds
                cat_map = self.categorical_maps[i]['map']
                n_categories = len(cat_map)
                cat_idx = max(0, min(cat_idx, n_categories - 1))
                
                result[name] = cat_map[cat_idx]
            else:
                # Continuous parameter
                value = x[i]
                if hasattr(value, "__len__"):
                    result[name] = float(value[0])
                else:
                    result[name] = float(value)
            i += 1
        
        return result
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a single solution or batch of solutions"""
        # If x is 1D, it's a single solution
        if len(x.shape) == 1:
            # Convert to parameter dictionary
            params = self._convert_to_param_dict(x)
            
            # Evaluate
            result = self.test_problem.evaluate(params)
            out["F"] = np.array(result)
        # If x is 2D, it's a batch of solutions
        else:
            # Convert each solution to parameter dictionary and evaluate
            results = []
            for xi in x:
                params = self._convert_to_param_dict(xi)
                result = self.test_problem.evaluate(params)
                results.append(result)
            
            # Stack results into a single array
            out["F"] = np.array(results)


def run_optimizer_on_problem(
    optimizer_class: Type,
    problem: TestProblem,
    budget: int,
    pop_size: int = 100,
    n_gen: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a pymoo optimizer on a test problem
    
    :param optimizer_class: The pymoo optimizer class to use
    :param problem: The test problem to optimize
    :param budget: The evaluation budget
    :param pop_size: Population size for the algorithm
    :param n_gen: Number of generations (calculated from budget if None)
    :param kwargs: Additional keyword arguments for the optimizer
    :return: Dictionary with results
    """
    # Create pymoo problem
    pymoo_problem = PymooWrappedProblem(problem)
    
    # Calculate number of generations from budget if not provided
    if n_gen is None:
        n_gen = max(1, budget // pop_size)
    
    # Handle Bayesian optimization algorithms differently
    if optimizer_class in [BUCB, QEHVI]:
        # Initialize our Bayesian optimizer with PyMOO wrapper
        optimizer = PymooBayesianWrapper(
            optimizer_class=optimizer_class,
            parameter_space=problem.get_parameter_space(),
            n_objectives=problem.num_objectives,
            batch_size=pop_size,
            **kwargs
        )
        
        # For Bayesian optimization, we use different termination
        # As they're sequential in nature with batch size = 1
        start_time = time.time()
        
        result = minimize(
            pymoo_problem,
            optimizer,
            ('n_eval', budget),
            seed=42,
            verbose=True
        )
        
        runtime = time.time() - start_time
        
        # Get the Pareto front
        pareto_xs = []
        for i in range(len(result.X)):
            params = pymoo_problem._convert_to_param_dict(result.X[i])
            pareto_xs.append(params)
        
        # Prepare results
        results = {
            'optimizer_name': optimizer_class.__name__,
            'problem_name': problem.name,
            'pareto_xs': pareto_xs,
            'pareto_ys': result.F.tolist(),
            'runtime': runtime,
            'budget': budget,
            'pop_size': pop_size,
            'n_gen': budget,  # For Bayesian methods, n_gen effectively equals budget
            'pymoo_result': result
        }
        
        return results
    else:
        # Create optimizer for evolutionary algorithms
        if optimizer_class == NSGA3:
            # NSGA-III needs reference directions
            ref_dirs = get_reference_directions(
                "das-dennis", problem.num_objectives, n_partitions=12)
            optimizer = optimizer_class(
                pop_size=pop_size,
                ref_dirs=ref_dirs,
                **kwargs
            )
        elif optimizer_class == MOEAD:
            # MOEA/D also needs reference directions
            n_partitions = 12
            if problem.num_objectives == 3:
                n_partitions = 10
            elif problem.num_objectives > 3:
                n_partitions = 6
                
            ref_dirs = get_reference_directions(
                "das-dennis", problem.num_objectives, n_partitions=n_partitions)
            
            # The number of reference directions determines the pop_size
            actual_pop_size = len(ref_dirs)
            print(f"Note: Using population size of {actual_pop_size} for MOEA/D based on reference directions")
            
            optimizer = optimizer_class(
                ref_dirs=ref_dirs,
                n_neighbors=15,
                **kwargs
            )
        else:
            optimizer = optimizer_class(
                pop_size=pop_size,
                **kwargs
            )
        
        # Run optimization
        start_time = time.time()
        result = minimize(
            pymoo_problem,
            optimizer,
            ('n_gen', n_gen),
            seed=42,
            verbose=True
        )
        runtime = time.time() - start_time
        
        # Convert results to parameter dictionaries
        pareto_xs = []
        for x in result.X:
            params = pymoo_problem._convert_to_param_dict(x)
            pareto_xs.append(params)
        
        # Prepare results
        results = {
            'optimizer_name': optimizer_class.__name__,
            'problem_name': problem.name,
            'pareto_xs': pareto_xs,
            'pareto_ys': result.F.tolist(),
            'runtime': runtime,
            'budget': budget,
            'pop_size': pop_size,
            'n_gen': n_gen,
            'pymoo_result': result
        }
        
        return results


def generate_report(results: Dict[str, Any], output_dir: str = None) -> None:
    """
    Generate a report from benchmark results
    
    :param results: Dictionary with benchmark results
    :param output_dir: Directory to save plots (if None, will show plots)
    """
    problem_name = results['problem_name']
    optimizer_name = results['optimizer_name']
    pareto_ys = np.array(results['pareto_ys'])
    runtime = results['runtime']
    
    print(f"\n{'='*80}")
    print(f"Benchmark Report: {optimizer_name} on {problem_name}")
    print(f"{'='*80}")
    print(f"Total time: {runtime:.2f} seconds")
    print(f"Pareto front size: {len(pareto_ys)}")
    
    # For 2D problems, plot the Pareto front
    if pareto_ys.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(pareto_ys[:, 0], pareto_ys[:, 1], color='red', label='Pareto front')
        
        # Sort Pareto points by first objective for line
        sorted_pareto = pareto_ys[pareto_ys[:, 0].argsort()]
        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--')
        
        plt.title(f'Pareto Front: {optimizer_name} on {problem_name}')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_pareto.png'))
        else:
            plt.show()
    
    # For 3D problems, create 3D plot if it's a 3-objective problem
    elif pareto_ys.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2], c='r', marker='o')
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(f'Pareto Front: {optimizer_name} on {problem_name}')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_pareto3d.png'))
        else:
            plt.show()


def compare_optimizers(
    problem: TestProblem,
    budget: int,
    pop_size: int = 100,
    output_dir: str = None
) -> None:
    """
    Compare all optimizers on a test problem
    
    :param problem: The test problem to optimize
    :param budget: The evaluation budget
    :param pop_size: Population size for the algorithms
    :param output_dir: Directory to save plots (if None, will show plots)
    """
    results_list = []
    
    for optimizer_name, optimizer_class in OPTIMIZERS.items():
        print(f"Running {optimizer_name} on {problem.name}...")
        results = run_optimizer_on_problem(
            optimizer_class=optimizer_class,
            problem=problem,
            budget=budget,
            pop_size=pop_size
        )
        results_list.append(results)
        
        # Generate individual report
        generate_report(results, output_dir)
    
    # Compare Pareto fronts for 2D problems
    if problem.num_objectives == 2:
        plt.figure(figsize=(10, 6))
        
        for results in results_list:
            pareto_ys = np.array(results['pareto_ys'])
            plt.scatter(pareto_ys[:, 0], pareto_ys[:, 1], label=f"{results['optimizer_name']} Pareto")
        
        plt.title(f'Pareto Front Comparison on {problem.name}')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'pareto_comparison_{problem.name}.png'))
        else:
            plt.show()
    
    # Compare runtimes
    optimizer_names = [r['optimizer_name'] for r in results_list]
    runtimes = [r['runtime'] for r in results_list]
    
    plt.figure(figsize=(10, 6))
    plt.bar(optimizer_names, runtimes)
    plt.title(f'Runtime Comparison on {problem.name}')
    plt.xlabel('Optimizer')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True, axis='y')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'runtime_comparison_{problem.name}.png'))
    else:
        plt.show()
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"Optimizer Comparison on {problem.name}")
    print(f"{'='*80}")
    print(f"{'Optimizer':<20} {'Runtime (s)':<12} {'Pareto Size':<12}")
    print(f"{'-'*44}")
    
    for results in results_list:
        optimizer_name = results['optimizer_name']
        runtime = results['runtime']
        pareto_size = len(results['pareto_ys'])
        
        print(f"{optimizer_name:<20} {runtime:<12.2f} {pareto_size:<12}")


def main():
    try:
        parser = argparse.ArgumentParser(description='Run benchmark experiments using off-the-shelf multi-objective optimization algorithms')
        parser.add_argument('--problem', type=str, help='Test problem to optimize')
        parser.add_argument('--optimizer', type=str, help='Optimizer to use')
        parser.add_argument('--budget', type=int, default=1000, help='Evaluation budget')
        parser.add_argument('--pop-size', type=int, default=100, help='Population size for the algorithm')
        parser.add_argument('--output-dir', type=str, help='Directory to save plots')
        parser.add_argument('--list-problems', action='store_true', help='List available test problems')
        parser.add_argument('--list-optimizers', action='store_true', help='List available optimizers')
        parser.add_argument('--compare', action='store_true', help='Compare all optimizers')
        
        args = parser.parse_args()
        
        # List available problems and optimizers if requested
        if args.list_problems:
            print("Available test problems:")
            for problem in list_test_problems():
                print(f"  {problem}")
            return
        
        if args.list_optimizers:
            print("Available optimizers:")
            for optimizer in OPTIMIZERS:
                print(f"  {optimizer}")
            return
        
        # Check if problem is specified
        if not args.problem:
            parser.error("Please specify a test problem with --problem")
        
        problem = get_test_problem(args.problem)
        
        # Compare all optimizers or run a single optimizer
        if args.compare:
            compare_optimizers(
                problem=problem,
                budget=args.budget,
                pop_size=args.pop_size,
                output_dir=args.output_dir
            )
        elif args.optimizer:
            # Map optimizer name to class
            if args.optimizer == 'bucb':
                optimizer_class = BUCB
            elif args.optimizer == 'qehvi':
                optimizer_class = QEHVI
            elif args.optimizer in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[args.optimizer]
            else:
                parser.error(f"Unknown optimizer: {args.optimizer}. Available optimizers: {list(OPTIMIZERS.keys())}")
            
            results = run_optimizer_on_problem(
                optimizer_class=optimizer_class,
                problem=problem,
                budget=args.budget,
                pop_size=args.pop_size
            )
            
            generate_report(results, args.output_dir)
        else:
            parser.error("Please specify an optimizer with --optimizer or use --compare to compare all optimizers")
    except Exception as e:
        import traceback
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main() 