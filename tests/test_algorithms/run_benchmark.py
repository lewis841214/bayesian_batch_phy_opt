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

from src.core.algorithm import MultiObjectiveOptimizer, SingleObjectiveOptimizer
from src.core.metrics import get_pareto_front, calculate_hypervolume
from src.algorithms.bayesian.q_ehvi import QEHVI
from src.algorithms.bayesian.q_parego import QParEGO
from src.algorithms.bayesian.bucb import BUCB
from src.algorithms.evolutionary.nsga2 import NSGAII
from src.algorithms.evolutionary.moead import MOEAD

from tests.test_algorithms.test_problems import TestProblem, get_test_problem, list_test_problems


# Dictionary of available optimization algorithms
OPTIMIZERS = {
    'q-ehvi': QEHVI,
    'q-parego': QParEGO,
    'bucb': BUCB,
    'nsga-ii': NSGAII,
    'moea-d': MOEAD,
}


def run_optimizer_on_problem(
    optimizer_class: Type[MultiObjectiveOptimizer],
    problem: TestProblem,
    budget: int,
    batch_size: int,
    optimizer_kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run an optimizer on a test problem
    
    :param optimizer_class: The optimizer class to use
    :param problem: The test problem to optimize
    :param budget: The evaluation budget
    :param batch_size: The batch size for evaluations
    :param optimizer_kwargs: Additional keyword arguments for the optimizer
    :return: Dictionary with results
    """
    optimizer_kwargs = optimizer_kwargs or {}
    
    # Get parameter space from problem
    param_space = problem.get_parameter_space()
    
    # Create optimizer
    optimizer = optimizer_class(
        parameter_space=param_space,
        budget=budget,
        batch_size=batch_size,
        n_objectives=problem.num_objectives,
        **optimizer_kwargs
    )
    
    # Track data
    all_xs = []
    all_ys = []
    hypervolumes = []
    times = []
    
    # Get reference point for hypervolume calculation
    ref_point = np.array(problem.get_reference_point())
    
    # Run optimization
    start_time_total = time.time()
    remaining_budget = budget
    
    while remaining_budget > 0:
        batch_size_curr = min(batch_size, remaining_budget)
        
        # Time the ask step
        start_time = time.time()
        xs = optimizer.ask(n=batch_size_curr)
        ask_time = time.time() - start_time
        
        # Evaluate points
        ys = [problem.evaluate(x) for x in xs]
        
        # Store data
        all_xs.extend(xs)
        all_ys.extend(ys)
        
        # Update optimizer and time the tell step
        start_time = time.time()
        optimizer.tell(xs, ys)
        tell_time = time.time() - start_time
        
        # Calculate hypervolume if we have enough objectives (2+)
        if problem.num_objectives >= 2:
            # Convert to numpy array
            ys_array = np.array(all_ys)
            
            # Find Pareto front
            pareto_front = get_pareto_front(ys_array)
            
            # Calculate hypervolume (for 2D case it's simple, otherwise use specialized function)
            try:
                hv = calculate_hypervolume(pareto_front, ref_point)
                # Handle both tensor and float values
                if hasattr(hv, 'item'):
                    hv = hv.item()
                hypervolumes.append(float(hv))
            except Exception as e:
                print(f"Warning: Error in hypervolume calculation: {e}")
                # Append last value or 0 if no previous values
                if hypervolumes:
                    hypervolumes.append(hypervolumes[-1])
                else:
                    hypervolumes.append(0.0)
        
        # Track time
        times.append((ask_time, tell_time))
        
        remaining_budget -= batch_size_curr
    
    # Get final Pareto front
    total_time = time.time() - start_time_total
    pareto_xs, pareto_ys = optimizer.recommend()
    
    # Prepare results
    results = {
        'optimizer_name': optimizer_class.__name__,
        'problem_name': problem.name,
        'all_xs': all_xs,
        'all_ys': all_ys,
        'pareto_xs': pareto_xs,
        'pareto_ys': pareto_ys,
        'hypervolumes': hypervolumes,
        'times': times,
        'total_time': total_time,
        'budget': budget,
        'batch_size': batch_size
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
    all_ys = np.array(results['all_ys'])
    pareto_ys = np.array(results['pareto_ys'])
    hypervolumes = results['hypervolumes']
    total_time = results['total_time']
    
    print(f"\n{'='*80}")
    print(f"Benchmark Report: {optimizer_name} on {problem_name}")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total evaluations: {len(all_ys)}")
    print(f"Pareto front size: {len(pareto_ys)}")
    
    if hypervolumes:
        print(f"Final hypervolume: {hypervolumes[-1]:.4f}")
    
    # Check if we have multi-objective results (at least 2 objectives)
    is_multi_objective = all_ys.shape[1] >= 2 if len(all_ys.shape) > 1 else False
    
    # For 2D problems, plot the Pareto front
    if is_multi_objective and all_ys.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_ys[:, 0], all_ys[:, 1], alpha=0.5, label='All points')
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
    elif not is_multi_objective:
        # For single objective, plot objective value over iterations
        plt.figure(figsize=(10, 6))
        iterations = list(range(1, len(all_ys) + 1))
        
        # Reshape all_ys if needed
        obj_values = all_ys.flatten() if len(all_ys.shape) > 1 else all_ys
        
        plt.plot(iterations, obj_values, marker='o')
        plt.axhline(y=min(obj_values), color='r', linestyle='--', label=f'Best: {min(obj_values):.4f}')
        
        plt.title(f'Objective Value: {optimizer_name} on {problem_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_objective.png'))
        else:
            plt.show()
    
    # Plot hypervolume progression
    if hypervolumes:
        plt.figure(figsize=(10, 6))
        iterations = list(range(1, len(hypervolumes) + 1))
        plt.plot(iterations, hypervolumes, marker='o')
        
        plt.title(f'Hypervolume Progression: {optimizer_name} on {problem_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_hypervolume.png'))
        else:
            plt.show()


def compare_optimizers(
    problem: TestProblem,
    optimizers: List[Type[MultiObjectiveOptimizer]],
    budget: int,
    batch_size: int,
    optimizer_kwargs: Dict[str, Dict[str, Any]] = None,
    output_dir: str = None
) -> None:
    """
    Compare multiple optimizers on a test problem
    
    :param problem: The test problem to optimize
    :param optimizers: List of optimizer classes to compare
    :param budget: The evaluation budget
    :param batch_size: The batch size for evaluations
    :param optimizer_kwargs: Dictionary mapping optimizer names to their kwargs
    :param output_dir: Directory to save plots (if None, will show plots)
    """
    optimizer_kwargs = optimizer_kwargs or {}
    results_list = []
    
    for optimizer_class in optimizers:
        kwargs = optimizer_kwargs.get(optimizer_class.__name__, {})
        results = run_optimizer_on_problem(
            optimizer_class=optimizer_class,
            problem=problem,
            budget=budget,
            batch_size=batch_size,
            optimizer_kwargs=kwargs
        )
        results_list.append(results)
        
        # Generate individual report
        generate_report(results, output_dir)
    
    # Compare hypervolumes if applicable
    if all(len(r['hypervolumes']) > 0 for r in results_list):
        plt.figure(figsize=(10, 6))
        
        for results in results_list:
            hypervolumes = results['hypervolumes']
            iterations = list(range(1, len(hypervolumes) + 1))
            plt.plot(iterations, hypervolumes, marker='o', label=results['optimizer_name'])
        
        plt.title(f'Hypervolume Comparison on {problem.name}')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'hypervolume_comparison_{problem.name}.png'))
        else:
            plt.show()
    
    # Check if all are multi-objective or single-objective
    all_multi = all(len(np.array(r['all_ys']).shape) > 1 and np.array(r['all_ys']).shape[1] >= 2 
                   for r in results_list)
    all_single = all(len(np.array(r['all_ys']).shape) == 1 or np.array(r['all_ys']).shape[1] == 1 
                    for r in results_list)
                    
    # Compare final Pareto fronts for 2D multi-objective problems
    if all_multi and problem.num_objectives == 2:
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
    # For single-objective problems, compare best values found
    elif all_single:
        plt.figure(figsize=(10, 6))
        
        optimizer_names = []
        best_values = []
        
        for results in results_list:
            optimizer_names.append(results['optimizer_name'])
            
            # Get best value
            all_ys = np.array(results['all_ys'])
            if len(all_ys.shape) > 1:
                best_value = np.min(all_ys[:, 0])
            else:
                best_value = np.min(all_ys)
                
            best_values.append(best_value)
        
        # Create bar chart of best values
        plt.bar(optimizer_names, best_values)
        plt.title(f'Best Values Found on {problem.name}')
        plt.xlabel('Optimizer')
        plt.ylabel('Best Value (Lower is Better)')
        plt.grid(True, axis='y')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'best_comparison_{problem.name}.png'))
        else:
            plt.show()
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"Optimizer Comparison on {problem.name}")
    print(f"{'='*80}")
    print(f"{'Optimizer':<20} {'Time (s)':<10} {'Pareto Size':<12} {'Final HV':<10}")
    print(f"{'-'*50}")
    
    for results in results_list:
        optimizer_name = results['optimizer_name']
        total_time = results['total_time']
        pareto_size = len(results['pareto_ys'])
        final_hv = results['hypervolumes'][-1] if results['hypervolumes'] else "N/A"
        
        print(f"{optimizer_name:<20} {total_time:<10.2f} {pareto_size:<12} {final_hv if isinstance(final_hv, str) else f'{final_hv:.4f}':<10}")


def main():
    try:
        print("Starting run_benchmark.py")
        parser = argparse.ArgumentParser(description='Run benchmark experiments for multi-objective optimization algorithms')
        parser.add_argument('--problem', type=str, help='Test problem to optimize')
        parser.add_argument('--optimizer', type=str, help='Optimizer to use')
        parser.add_argument('--budget', type=int, default=100, help='Evaluation budget')
        parser.add_argument('--batch-size', type=int, default=5, help='Batch size for evaluations')
        parser.add_argument('--output-dir', type=str, help='Directory to save plots')
        parser.add_argument('--list-problems', action='store_true', help='List available test problems')
        parser.add_argument('--list-optimizers', action='store_true', help='List available optimizers')
        parser.add_argument('--compare', action='store_true', help='Compare all optimizers')
        
        args = parser.parse_args()
        print(f"Parsed arguments: {args}")
        
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
        
        # Check if problem and optimizer are specified
        if args.compare:
            # Run comparison of all optimizers on specified problem
            if not args.problem:
                parser.error("Please specify a test problem with --problem")
            
            problem = get_test_problem(args.problem)
            optimizers = list(OPTIMIZERS.values())
            
            compare_optimizers(
                problem=problem,
                optimizers=optimizers,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
        elif args.problem and args.optimizer:
            # Run single optimizer on problem
            problem = get_test_problem(args.problem)
            
            if args.optimizer not in OPTIMIZERS:
                parser.error(f"Unknown optimizer: {args.optimizer}. Available optimizers: {list(OPTIMIZERS.keys())}")
            
            optimizer_class = OPTIMIZERS[args.optimizer]
            
            results = run_optimizer_on_problem(
                optimizer_class=optimizer_class,
                problem=problem,
                budget=args.budget,
                batch_size=args.batch_size
            )
            
            generate_report(results, args.output_dir)
        else:
            parser.print_help()
    except Exception as e:
        import traceback
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main() 