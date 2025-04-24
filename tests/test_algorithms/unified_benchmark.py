#!/usr/bin/env python3
import argparse
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Type

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test problems
from tests.test_algorithms.test_problems import get_test_problem, list_test_problems

# Import algorithm adapters
from src.adapters.algorithm_adapters import get_algorithm_adapter

def run_optimization(algorithm_name, problem_name, budget, batch_size, output_dir=None):
    """Run specified optimization algorithm on the given problem"""
    print(f"Running {algorithm_name} on {problem_name} with budget {budget} and batch size {batch_size}")
    
    # Get test problem
    problem = get_test_problem(problem_name)
    
    # Get algorithm adapter
    adapter = get_algorithm_adapter(algorithm_name)
    adapter.setup(problem, budget, batch_size)
    
    # Run optimization
    start_time = time.time()
    
    remaining_budget = budget
    while remaining_budget > 0:
        # Determine batch size for this iteration
        current_batch = min(batch_size, remaining_budget)
        
        # Get candidates
        candidates = adapter.ask()
        
        # Evaluate candidates
        values = []
        for candidate in candidates[:current_batch]:  # Ensure we respect batch size
            value = problem.evaluate(candidate)
            values.append(value)
        
        # Update algorithm
        adapter.tell(candidates[:current_batch], values)
        
        # Update budget
        remaining_budget -= current_batch
    
    # Get final results
    pareto_x, pareto_y = adapter.get_result()
    hypervolume = adapter.get_hypervolume()
    
    runtime = time.time() - start_time
    print(f"Optimization completed in {runtime:.2f} seconds")
    print(f"Final hypervolume: {hypervolume:.6f}")
    print(f"Pareto front size: {len(pareto_x)}")
    
    # Plot results
    if output_dir and problem.num_objectives in [2, 3]:
        pareto_ys = np.array(pareto_y)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if problem.num_objectives == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(pareto_ys[:, 0], pareto_ys[:, 1], color='red', label='Pareto front')
            
            # Sort for line if more than one point
            if len(pareto_ys) > 1:
                sorted_pareto = pareto_ys[pareto_ys[:, 0].argsort()]
                plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--')
            
            plt.title(f'Pareto Front: {algorithm_name} on {problem_name}')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto.png'))
            plt.close()
        
        elif problem.num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2], c='r', marker='o')
            
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title(f'Pareto Front: {algorithm_name} on {problem_name}')
            
            plt.savefig(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto3d.png'))
            plt.close()
    
    return {
        'algorithm': algorithm_name,
        'problem': problem_name,
        'runtime': runtime,
        'hypervolume': hypervolume,
        'pareto_size': len(pareto_x),
        'pareto_y': pareto_y
    }

def compare_algorithms(problem_name, algorithms, budget, batch_size, output_dir=None):
    """Compare specified algorithms on a problem"""
    results = []
    
    for algorithm in algorithms:
        result = run_optimization(
            algorithm_name=algorithm,
            problem_name=problem_name,
            budget=budget,
            batch_size=batch_size,
            output_dir=output_dir
        )
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"Comparison on {problem_name}")
    print("="*80)
    print(f"{'Algorithm':<10} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['algorithm']:<10} {result['runtime']:<15.2f} {result['hypervolume']:<15.6f} {result['pareto_size']:<15}")
    
    # Plot comparisons
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Bar chart for runtimes
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r['algorithm'] for r in results],
            [r['runtime'] for r in results]
        )
        plt.title(f'Runtime Comparison on {problem_name}')
        plt.xlabel('Algorithm')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'runtime_comparison_{problem_name}.png'))
        plt.close()
        
        # Bar chart for hypervolumes
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r['algorithm'] for r in results],
            [r['hypervolume'] for r in results]
        )
        plt.title(f'Hypervolume Comparison on {problem_name}')
        plt.xlabel('Algorithm')
        plt.ylabel('Hypervolume')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'hypervolume_comparison_{problem_name}.png'))
        plt.close()
        
        # If 2-objective problem, plot all Pareto fronts
        problem = get_test_problem(problem_name)
        if problem.num_objectives == 2:
            plt.figure(figsize=(10, 6))
            
            for result in results:
                pareto_y = np.array(result['pareto_y'])
                if len(pareto_y) > 0:
                    plt.scatter(pareto_y[:, 0], pareto_y[:, 1], label=result['algorithm'])
                    if len(pareto_y) > 1:
                        sorted_pareto = pareto_y[pareto_y[:, 0].argsort()]
                        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], '--')
            
            plt.title(f'Pareto Front Comparison on {problem_name}')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'pareto_comparison_{problem_name}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Unified benchmark for optimization algorithms')
    parser.add_argument('--problem', type=str, default='constrained', help='Test problem to optimize')
    parser.add_argument('--algorithm', type=str, help='Specific algorithm to use (qnehvi, qehvi, qparego, nsga2, moead, nsga3)')
    parser.add_argument('--budget', type=int, default=50, help='Evaluation budget')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='output/unified_benchmark', help='Directory to save plots')
    parser.add_argument('--list-problems', action='store_true', help='List available test problems')
    parser.add_argument('--compare', action='store_true', help='Compare multiple algorithms')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['qnehvi', 'nsga2'], 
                        help='Algorithms to compare (when using --compare)')
    
    args = parser.parse_args()
    
    # List available problems if requested
    if args.list_problems:
        print("Available test problems:")
        for problem in list_test_problems():
            print(f"  {problem}")
        return
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison or single algorithm
    if args.compare:
        compare_algorithms(
            problem_name=args.problem,
            algorithms=args.algorithms,
            budget=args.budget,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    elif args.algorithm:
        run_optimization(
            algorithm_name=args.algorithm,
            problem_name=args.problem,
            budget=args.budget,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    else:
        print("Please specify an algorithm with --algorithm or use --compare to compare algorithms")

if __name__ == '__main__':
    main() 