#!/usr/bin/env python3
import argparse
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test problems
from tests.test_algorithms.test_problems import get_test_problem, list_test_problems

# Import Bayesian optimization algorithms
from src.algorithms.bayesian.qnehvi import QNEHVI
from src.algorithms.bayesian.qehvi import QEHVI
from src.algorithms.bayesian.qparego import QNParEGO

def run_optimizer_on_problem(optimizer_name, problem_name, budget, batch_size, output_dir=None):
    """Run a specified optimizer on a test problem with the given budget"""
    print(f"Running {optimizer_name} on {problem_name} with budget {budget} and batch size {batch_size}")
    
    # Get the test problem
    problem = get_test_problem(problem_name)
    
    # Get parameter space from the problem
    parameter_space = problem.get_parameter_space()
    
    # Set the reference point
    ref_point = problem.get_reference_point()
    
    # Initialize the optimizer based on the name
    if optimizer_name.lower() == 'qnehvi':
        optimizer = QNEHVI(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128
        )
    elif optimizer_name.lower() == 'qehvi':
        optimizer = QEHVI(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128
        )
    elif optimizer_name.lower() == 'qparego':
        optimizer = QNParEGO(
            parameter_space=parameter_space,
            budget=budget,
            batch_size=batch_size,
            n_objectives=problem.num_objectives,
            ref_point=ref_point,
            mc_samples=128
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Run optimization process manually
    start_time = time.time()
    
    remaining_budget = budget
    while remaining_budget > 0:
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, remaining_budget)
        
        # Get batch of points to evaluate
        batch_params = optimizer.ask(n=current_batch_size)
        
        # Evaluate points
        batch_values = []
        for params in batch_params:
            values = problem.evaluate(params)
            batch_values.append(values)
        
        # Update optimizer with new observations
        optimizer.tell(batch_params, batch_values)
        
        # Update budget
        remaining_budget -= current_batch_size
    
    # Get final results
    pareto_params, pareto_values = optimizer.recommend()
    
    # Calculate hypervolume
    hypervolume = optimizer.get_hypervolume()
    
    # Report results
    runtime = time.time() - start_time
    print(f"Optimization completed in {runtime:.2f} seconds")
    print(f"Final hypervolume: {hypervolume:.6f}")
    print(f"Pareto front size: {len(pareto_params)}")
    
    # Plot results if output directory is provided
    if output_dir and problem.num_objectives in [2, 3]:
        pareto_ys = np.array(pareto_values)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if problem.num_objectives == 2:
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
            
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_pareto.png'))
            plt.close()
        
        elif problem.num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2], c='r', marker='o')
            
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title(f'Pareto Front: {optimizer_name} on {problem_name}')
            
            plt.savefig(os.path.join(output_dir, f'{optimizer_name}_{problem_name}_pareto3d.png'))
            plt.close()
    
    return {
        'optimizer': optimizer_name,
        'problem': problem_name,
        'runtime': runtime,
        'hypervolume': hypervolume,
        'pareto_size': len(pareto_params),
        'pareto_values': pareto_values
    }

def compare_optimizers(problem_name, budget, batch_size, output_dir=None):
    """Compare all Bayesian optimization algorithms on a specific problem"""
    results = []
    
    for optimizer_name in ['qnehvi', 'qehvi', 'qparego']:
        result = run_optimizer_on_problem(
            optimizer_name=optimizer_name,
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
    print(f"{'Optimizer':<10} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['optimizer']:<10} {result['runtime']:<15.2f} {result['hypervolume']:<15.6f} {result['pareto_size']:<15}")
    
    # Plot comparison if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Bar chart for runtimes
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r['optimizer'] for r in results],
            [r['runtime'] for r in results]
        )
        plt.title(f'Runtime Comparison on {problem_name}')
        plt.xlabel('Optimizer')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'runtime_comparison_{problem_name}.png'))
        plt.close()
        
        # Bar chart for hypervolumes
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r['optimizer'] for r in results],
            [r['hypervolume'] for r in results]
        )
        plt.title(f'Hypervolume Comparison on {problem_name}')
        plt.xlabel('Optimizer')
        plt.ylabel('Hypervolume')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'hypervolume_comparison_{problem_name}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test Bayesian Optimization algorithms on test problems')
    parser.add_argument('--problem', type=str, default='mixed', help='Test problem to optimize')
    parser.add_argument('--optimizer', type=str, help='Specific optimizer to use (qnehvi, qehvi, qparego)')
    parser.add_argument('--budget', type=int, default=50, help='Evaluation budget')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='output/bo_test', help='Directory to save plots')
    parser.add_argument('--list-problems', action='store_true', help='List available test problems')
    parser.add_argument('--compare', action='store_true', help='Compare all optimizers')
    
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
    
    # Run comparison or single optimizer
    if args.compare:
        compare_optimizers(
            problem_name=args.problem,
            budget=args.budget,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    elif args.optimizer:
        run_optimizer_on_problem(
            optimizer_name=args.optimizer,
            problem_name=args.problem,
            budget=args.budget,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    else:
        print("Please specify an optimizer with --optimizer or use --compare to compare all optimizers")

if __name__ == '__main__':
    main() 