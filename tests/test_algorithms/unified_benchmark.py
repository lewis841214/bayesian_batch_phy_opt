#!/usr/bin/env python3
import argparse
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Type
from tqdm import tqdm

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
    all_evaluated_x = []
    all_evaluated_y = []
    
    # Create progress bar for overall optimization
    pbar = tqdm(total=budget, desc=f"Optimizing with {algorithm_name}", 
                unit="evaluations", leave=True)
    
    iteration = 0
    while remaining_budget > 0:
        # Determine batch size for this iteration
        current_batch = min(batch_size, remaining_budget)
        
        # Get candidates
        tqdm.write(f"\nIteration {iteration+1}: Requesting {current_batch} candidates...")
        candidates = adapter.ask()
        
        # Evaluate candidates
        values = []
        for i, candidate in enumerate(tqdm(candidates[:current_batch], desc="Evaluating candidates", leave=False)):
            value = problem.evaluate(candidate)
            values.append(value)
            
            # Store all evaluated points
            all_evaluated_x.append(candidate)
            all_evaluated_y.append(value)
        
        # Update algorithm
        tqdm.write(f"Updating model with {current_batch} evaluations...")
        adapter.tell(candidates[:current_batch], values)
        
        # Update budget and progress bar
        remaining_budget -= current_batch
        pbar.update(current_batch)
        
        # Print some info about current best point
        if hasattr(adapter, 'get_hypervolume'):
            tqdm.write(f"Current hypervolume: {adapter.get_hypervolume():.6f}")
        
        iteration += 1
    
    pbar.close()
    
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
        all_ys = np.array(all_evaluated_y)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if problem.num_objectives == 2:
            # Create figure with proper layout for colorbar
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all evaluated points with alpha scaling by recency
            if len(all_ys) > 0:
                # Use colormap to show progression of evaluations
                n_points = len(all_ys)
                
                # Create a scatter plot with a color gradient
                scatter = ax.scatter(all_ys[:, 0], all_ys[:, 1], 
                                    c=np.arange(n_points),
                                    cmap='Blues',
                                    alpha=0.7, 
                                    label='Evaluated points')
                
                # Add a color bar to show progression (using the figure and specific axes)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Evaluation order')
            
            # Plot Pareto front with higher opacity
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], color='red', label='Pareto front')
            
            # Sort for line if more than one point
            if len(pareto_ys) > 1:
                sorted_pareto = pareto_ys[pareto_ys[:, 0].argsort()]
                ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--')
            
            ax.set_title(f'Pareto Front: {algorithm_name} on {problem_name}')
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto.png'))
            plt.close()
            
            # Save raw data for later analysis
            np.save(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto_y.npy'), pareto_ys)
            np.save(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_all_y.npy'), all_ys)
        
        elif problem.num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all evaluated points with color scaling by recency
            if len(all_ys) > 0:
                # Use colormap to show progression of evaluations
                n_points = len(all_ys)
                
                # Create a scatter plot with a color gradient
                scatter = ax.scatter(all_ys[:, 0], all_ys[:, 1], all_ys[:, 2], 
                                    c=np.arange(n_points), 
                                    cmap='Blues',
                                    alpha=0.7,
                                    label='Evaluated points')
                
                # Add a color bar to show progression
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Evaluation order')
            
            # Plot Pareto front with higher opacity
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2], c='r', marker='o', label='Pareto front')
            
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title(f'Pareto Front: {algorithm_name} on {problem_name}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto3d.png'))
            plt.close()
    
    return {
        'algorithm': algorithm_name,
        'problem': problem_name,
        'runtime': runtime,
        'hypervolume': hypervolume,
        'pareto_size': len(pareto_x),
        'pareto_y': pareto_y,
        'all_evaluated_x': all_evaluated_x,
        'all_evaluated_y': all_evaluated_y
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
    
    return results

def compare_across_problems(problems, algorithms, budget, batch_size, output_dir=None):
    """Compare algorithms across multiple problems"""
    all_results = []
    
    for problem in problems:
        # Run each algorithm separately on this problem instead of all at once
        problem_results = []
        for algorithm_name in algorithms:
            print(f"\nRunning {algorithm_name} on {problem}")
            # Get a fresh instance of the algorithm adapter for each problem-algorithm pair
            adapter = get_algorithm_adapter(algorithm_name)
            problem_obj = get_test_problem(problem)
            adapter.setup(problem_obj, budget, batch_size)
            
            # Run optimization for this algorithm
            result = run_optimization(
                algorithm_name=algorithm_name,
                problem_name=problem,
                budget=budget,
                batch_size=batch_size,
                output_dir=os.path.join(output_dir, problem) if output_dir else None
            )
            problem_results.append(result)
            
        # Print comparison table for this problem
        print("\n" + "="*80)
        print(f"Comparison on {problem}")
        print("="*80)
        print(f"{'Algorithm':<10} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15}")
        print("-"*80)
        
        for result in problem_results:
            print(f"{result['algorithm']:<10} {result['runtime']:<15.2f} {result['hypervolume']:<15.6f} {result['pareto_size']:<15}")
            
        # Add results to overall results
        all_results.extend(problem_results)
        
        # Plot comparisons for this problem
        if output_dir:
            problem_dir = os.path.join(output_dir, problem)
            os.makedirs(problem_dir, exist_ok=True)
            
            # Bar chart for runtimes
            plt.figure(figsize=(10, 6))
            plt.bar(
                [r['algorithm'] for r in problem_results],
                [r['runtime'] for r in problem_results]
            )
            plt.title(f'Runtime Comparison on {problem}')
            plt.xlabel('Algorithm')
            plt.ylabel('Runtime (seconds)')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(problem_dir, f'runtime_comparison_{problem}.png'))
            plt.close()
            
            # Bar chart for hypervolumes
            plt.figure(figsize=(10, 6))
            plt.bar(
                [r['algorithm'] for r in problem_results],
                [r['hypervolume'] for r in problem_results]
            )
            plt.title(f'Hypervolume Comparison on {problem}')
            plt.xlabel('Algorithm')
            plt.ylabel('Hypervolume')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(problem_dir, f'hypervolume_comparison_{problem}.png'))
            plt.close()
            
            # If 2-objective problem, plot all Pareto fronts
            problem_obj = get_test_problem(problem)
            if problem_obj.num_objectives == 2:
                plt.figure(figsize=(10, 6))
                
                for result in problem_results:
                    pareto_y = np.array(result['pareto_y'])
                    if len(pareto_y) > 0:
                        plt.scatter(pareto_y[:, 0], pareto_y[:, 1], label=result['algorithm'])
                        if len(pareto_y) > 1:
                            sorted_pareto = pareto_y[pareto_y[:, 0].argsort()]
                            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], '--')
                
                plt.title(f'Pareto Front Comparison on {problem}')
                plt.xlabel('Objective 1')
                plt.ylabel('Objective 2')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(problem_dir, f'pareto_comparison_{problem}.png'))
                plt.close()
    
    # Create summary table and plots across all problems
    if output_dir:
        # Create data for performance profiles
        # Organize data by algorithm and problem
        data_by_algo = {}
        metrics = ['runtime', 'hypervolume', 'pareto_size']
        
        for result in all_results:
            algo = result['algorithm']
            if algo not in data_by_algo:
                data_by_algo[algo] = {'problems': [], 'runtime': [], 'hypervolume': [], 'pareto_size': []}
            
            data_by_algo[algo]['problems'].append(result['problem'])
            data_by_algo[algo]['runtime'].append(result['runtime'])
            data_by_algo[algo]['hypervolume'].append(result['hypervolume'])
            data_by_algo[algo]['pareto_size'].append(result['pareto_size'])
        
        # Create comparison plots across problems
        # Average runtime by algorithm
        plt.figure(figsize=(12, 8))
        avg_runtimes = [np.mean(data_by_algo[algo]['runtime']) for algo in algorithms]
        plt.bar(algorithms, avg_runtimes)
        plt.title('Average Runtime Across All Problems')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Runtime (seconds)')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, 'avg_runtime_comparison.png'))
        plt.close()
        
        # Average hypervolume by algorithm 
        plt.figure(figsize=(12, 8))
        avg_hv = [np.mean(data_by_algo[algo]['hypervolume']) for algo in algorithms]
        plt.bar(algorithms, avg_hv)
        plt.title('Average Hypervolume Across All Problems')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Hypervolume')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, 'avg_hypervolume_comparison.png'))
        plt.close()
        
        # Create a heatmap comparing algorithms and problems
        if len(problems) > 1:
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                data = np.zeros((len(algorithms), len(problems)))
                
                for i, algo in enumerate(algorithms):
                    for j, problem in enumerate(problems):
                        # Find the result for this algorithm-problem pair
                        for result in all_results:
                            if result['algorithm'] == algo and result['problem'] == problem:
                                if metric == 'runtime':
                                    # For runtime, lower is better
                                    data[i, j] = -result[metric]
                                else:
                                    # For other metrics, higher is better
                                    data[i, j] = result[metric]
                                break
                
                plt.imshow(data, cmap='viridis')
                plt.colorbar(label=metric.capitalize())
                plt.xticks(np.arange(len(problems)), problems, rotation=45)
                plt.yticks(np.arange(len(algorithms)), algorithms)
                plt.title(f'{metric.capitalize()} Comparison')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'))
                plt.close()
        
        # Create summary table in a text file
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write("Performance Summary Across All Problems\n")
            f.write("="*80 + "\n\n")
            
            # Summary by algorithm
            f.write("Performance by Algorithm (averages)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Algorithm':<10} {'Avg Runtime (s)':<15} {'Avg Hypervolume':<20} {'Avg Pareto Size':<15}\n")
            
            for algo in algorithms:
                avg_runtime = np.mean(data_by_algo[algo]['runtime'])
                avg_hv = np.mean(data_by_algo[algo]['hypervolume'])
                avg_pareto_size = np.mean(data_by_algo[algo]['pareto_size'])
                
                f.write(f"{algo:<10} {avg_runtime:<15.2f} {avg_hv:<20.6f} {avg_pareto_size:<15.2f}\n")
            
            f.write("\n\n")
            
            # Individual results
            f.write("All Results\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Algorithm':<10} {'Problem':<15} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15}\n")
            
            for result in all_results:
                f.write(f"{result['algorithm']:<10} {result['problem']:<15} {result['runtime']:<15.2f} "
                        f"{result['hypervolume']:<15.6f} {result['pareto_size']:<15}\n")

def main():
    parser = argparse.ArgumentParser(description='Unified benchmark for optimization algorithms')
    parser.add_argument('--problem', type=str, default='constrained', help='Test problem to optimize')
    parser.add_argument('--problems', type=str, nargs='+', help='Multiple test problems to optimize')
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
        if args.problems:
            # Compare algorithms across multiple problems
            compare_across_problems(
                problems=args.problems,
                algorithms=args.algorithms,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
        else:
            # Compare algorithms on a single problem
            compare_algorithms(
                problem_name=args.problem,
                algorithms=args.algorithms,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
    elif args.algorithm:
        if args.problems:
            # Run a single algorithm on multiple problems
            for problem in args.problems:
                run_optimization(
                    algorithm_name=args.algorithm,
                    problem_name=problem,
                    budget=args.budget,
                    batch_size=args.batch_size,
                    output_dir=os.path.join(args.output_dir, problem) if args.output_dir else None
                )
        else:
            # Run a single algorithm on a single problem
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