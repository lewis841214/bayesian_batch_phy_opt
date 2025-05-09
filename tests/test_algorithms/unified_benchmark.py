#!/usr/bin/env python3
import argparse
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Type
from tqdm import tqdm
import datetime
# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test problems
from tests.test_algorithms.test_problems import get_test_problem, list_test_problems

# Import algorithm adapters
from src.adapters.algorithm_adapters import get_algorithm_adapter

def run_optimization(algorithm_name, problem_name, budget, batch_size, output_dir=None, verbose=False):
    """Run specified optimization algorithm on the given problem"""
    print(f"Running {algorithm_name} on {problem_name} with budget {budget} and batch size {batch_size}")
    
    # Get test problem
    problem = get_test_problem(problem_name)
    adapter = get_algorithm_adapter(algorithm_name)

    if 'hidden_map_dim' in problem.__dict__ and algorithm_name in ['nn-qnehvi', 'nnk-qnehvi']:
        adapter.setup(problem, budget, batch_size, hidden_map_dim=problem.hidden_map_dim)
    else:
        adapter.setup(problem, budget, batch_size)
    
    # Run optimization
    start_time = time.time()
    
    remaining_budget = budget
    all_evaluated_x = []
    all_evaluated_y = []
    hypervolume_history = []  # Track hypervolume at each iteration
    
    # Create progress bar for overall optimization
    pbar = tqdm(total=budget, desc=f"Optimizing with {algorithm_name}", 
                unit="evaluations", leave=True)
    
    iteration = 0
    while remaining_budget > 0:
        # Determine batch size for this iteration
        current_batch = min(batch_size, remaining_budget)
        
        # Get candidates
        tqdm.write(f"\nIteration {iteration+1}: Requesting {current_batch} candidates...")
        
        # Only pass output_dir to ask() for specific Bayesian optimization algorithms that support it
        # List of Bayesian optimization algorithms that use surrogate models and support output_dir
        supported_models = ['qnehvi', 'nn-qnehvi', 'nnk-qnehvi']
        
        try:
            if output_dir and algorithm_name.lower() in supported_models:
                # Create a specific directory for this run
                model_plots_dir = os.path.join(output_dir, f"model_plots_{algorithm_name}")
                os.makedirs(model_plots_dir, exist_ok=True)
                # Try to get candidates with model prediction plots
                candidates = adapter.ask(output_dir=model_plots_dir)
            else:
                # Get candidates without model prediction plots
                candidates = adapter.ask()
        except TypeError as e:
            # If the adapter doesn't support output_dir, fallback to standard ask
            if "unexpected keyword argument 'output_dir'" in str(e):
                tqdm.write(f"Warning: {algorithm_name} adapter doesn't support output_dir parameter")
                candidates = adapter.ask()
            else:
                # Re-raise if it's a different TypeError
                raise
        
        # Evaluate candidates
        values = []
        hidden_maps = []
        for i, candidate in enumerate(tqdm(candidates[:current_batch], desc="Evaluating candidates", leave=False)):
            value = problem.evaluate(candidate)
            if type(value) == tuple:
                if algorithm_name not in ['nn-qnehvi', 'nnk-qnehvi']:
                    value = value[0]
                else:
                    hidden_map= value[1]
                    value = value[0]
                    hidden_maps.append(hidden_map)

            values.append(value)
            
            
            # Store all evaluated points
            all_evaluated_x.append(candidate)
            all_evaluated_y.append(value)
        
        # Update algorithm
        tqdm.write(f"Updating model with {current_batch} evaluations...")
        if algorithm_name in ['nn-qnehvi', 'nnk-qnehvi'] and hidden_maps:
            adapter.tell(candidates[:current_batch], values, hidden_maps)
        else:
            adapter.tell(candidates[:current_batch], values)
        
        # Update budget and progress bar
        remaining_budget -= current_batch
        pbar.update(current_batch)
        
        # Record hypervolume
        current_hv = 0.0
        if hasattr(adapter, 'get_hypervolume'):
            current_hv = adapter.get_hypervolume()
            hypervolume_history.append(current_hv)
            tqdm.write(f"Iteration {iteration+1}: Current hypervolume: {current_hv:.6f}")
        
        iteration += 1
    
    pbar.close()
    
    # Get final results
    pareto_x, pareto_y = adapter.get_result()
    hypervolume = adapter.get_hypervolume()
    
    runtime = time.time() - start_time
    print(f"Optimization completed in {runtime:.2f} seconds")
    print(f"Final hypervolume: {hypervolume:.6f}")
    print(f"Pareto front size: {len(pareto_x)}")
    print(f"Total evaluated points: {len(all_evaluated_y)}")
    
    # Plot results
    if output_dir and problem.num_objectives in [2, 3]:
        pareto_ys = np.array(pareto_y)
        all_ys = np.array(all_evaluated_y)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot hypervolume progression
        if hypervolume_history:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(hypervolume_history) + 1), hypervolume_history, 
                     marker='o', markersize=4, linewidth=2)
            plt.title(f'Hypervolume Convergence: {algorithm_name} on {problem_name}')
            plt.xlabel('Iteration')
            plt.ylabel('Hypervolume')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_hypervolume.png'))
            plt.close()
        
        if problem.num_objectives == 2:
            # Create figure with proper layout for colorbar
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot all evaluated points with alpha scaling by recency
            if len(all_ys) > 0:
                # Use colormap to show progression of evaluations
                n_points = len(all_ys)
                
                # Create a scatter plot with a color gradient
                scatter = ax.scatter(all_ys[:, 0], all_ys[:, 1], 
                                    c=np.arange(n_points),
                                    cmap='Blues',
                                    alpha=0.7, 
                                    label=f'Evaluated points ({n_points})')
                
                # Add a color bar to show progression (using the figure and specific axes)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Evaluation order')
                
                # Add numbers to selected evaluated points
                # For clarity, number only a subset of points - every nth point
                step = max(1, n_points // 20)  # Show at most ~20 numbered points
                for i in range(0, n_points, step):
                    # Use a different format than Pareto points - E for evaluated
                    ax.annotate(f"E{i+1}", (all_ys[i, 0], all_ys[i, 1]), 
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=8, color='darkblue', weight='bold',
                               bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
            
            # Plot Pareto front with higher opacity
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], color='red', s=80, label=f'Pareto front ({len(pareto_ys)})')
            
            # Add point numbers to Pareto front points
            for i, (x, y) in enumerate(zip(pareto_ys[:, 0], pareto_ys[:, 1])):
                ax.annotate(f"P{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, fontweight='bold', color='black')
            
            # Add stats text
            stats_text = f"Total points: {len(all_ys)}\nPareto points: {len(pareto_ys)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8), va='top')
            
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

            # breakpoint()
            
            # Save raw data for later analysis
            np.save(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_pareto_y.npy'), pareto_ys)
            np.save(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_all_y.npy'), all_ys)
            np.save(os.path.join(output_dir, f'{algorithm_name}_{problem_name}_hypervolume.npy'), np.array(hypervolume_history))
        
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
                                    label=f'Evaluated points ({n_points})')
                
                # Add a color bar to show progression
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Evaluation order')
                
                # Number a subset of points (too many would be cluttered in 3D)
                step = max(1, n_points // 10)  # Show at most ~10 numbered points
                for i in range(0, n_points, step):
                    ax.text(all_ys[i, 0], all_ys[i, 1], all_ys[i, 2], f"E{i+1}", 
                           fontsize=8, color='darkblue', weight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, pad=0.1))
            
            # Plot Pareto front with higher opacity
            ax.scatter(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2], c='r', marker='o', s=80, 
                      label=f'Pareto front ({len(pareto_ys)})')
            
            # Add point numbers to Pareto front points
            for i, (x, y, z) in enumerate(zip(pareto_ys[:, 0], pareto_ys[:, 1], pareto_ys[:, 2])):
                ax.text(x, y, z, f"P{i+1}", fontsize=9, fontweight='bold', color='black')
            
            # Add stats text
            stats_text = f"Total points: {len(all_ys)}\nPareto points: {len(pareto_ys)}"
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8), va='top')
            
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
        'hypervolume_history': hypervolume_history,
        'pareto_size': len(pareto_x),
        'total_points': len(all_evaluated_y),
        'pareto_y': pareto_y,
        'all_evaluated_x': all_evaluated_x,
        'all_evaluated_y': all_evaluated_y
    }

def compare_algorithms(problem_name, algorithms, budget, batch_size, output_dir=None, verbose=False):
    """Compare specified algorithms on a problem"""
    results = []
    
    for algorithm in algorithms:
        result = run_optimization(
            algorithm_name=algorithm,
            problem_name=problem_name,
            budget=budget,
            batch_size=batch_size,
            output_dir=output_dir,
            verbose=verbose
        )
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"Comparison on {problem_name}")
    print("="*80)
    print(f"{'Algorithm':<10} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15} {'Total Points':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['algorithm']:<10} {result['runtime']:<15.2f} {result['hypervolume']:<15.6f} "
              f"{result['pareto_size']:<15} {result.get('total_points', 0):<15}")
    
    # Plot comparisons
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot hypervolume convergence for all algorithms
        plt.figure(figsize=(12, 8))
        for result in results:
            if 'hypervolume_history' in result and result['hypervolume_history']:
                plt.plot(
                    range(1, len(result['hypervolume_history']) + 1), 
                    result['hypervolume_history'],
                    label=result['algorithm'],
                    marker='o',
                    markersize=4,
                    linewidth=2
                )
                
        plt.title(f'Hypervolume Convergence on {problem_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hypervolume_convergence_{problem_name}.png'))
        plt.close()
        
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
            plt.figure(figsize=(12, 8))
            
            # Dictionary to store point counts
            point_counts = {}
            
            for result in results:
                pareto_y = np.array(result['pareto_y'])
                all_y = np.array(result['all_evaluated_y'])
                algo_name = result['algorithm']
                point_counts[algo_name] = {
                    'pareto': len(pareto_y),
                    'total': len(all_y)
                }
                
                # Plot all evaluated points with low opacity
                if len(all_y) > 0:
                    plt.scatter(all_y[:, 0], all_y[:, 1], alpha=0.3, s=20, 
                               label=f"{algo_name} all points ({len(all_y)})")
                
                # Plot Pareto front with higher opacity
                if len(pareto_y) > 0:
                    plt.scatter(pareto_y[:, 0], pareto_y[:, 1], alpha=0.8, s=80, 
                              label=f"{algo_name} Pareto ({len(pareto_y)})")
                    
                    if len(pareto_y) > 1:
                        sorted_pareto = pareto_y[pareto_y[:, 0].argsort()]
                        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], '--')
            
            # Add a summary of point counts in the top-left corner
            summary_text = "\n".join([f"{algo}: {counts['total']} total, {counts['pareto']} Pareto" 
                                    for algo, counts in point_counts.items()])
            plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
                    fontsize=10, fontweight='bold', verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(f'Pareto Front Comparison on {problem_name}')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'pareto_comparison_{problem_name}.png'))
            plt.close()
    
    return results

def compare_across_problems(problems, algorithms, budget, batch_size, output_dir=None, verbose=False):
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
                output_dir=os.path.join(output_dir, problem) if output_dir else None,
                verbose=verbose
            )
            problem_results.append(result)
            
        # Print comparison table for this problem
        print("\n" + "="*80)
        print(f"Comparison on {problem}")
        print("="*80)
        print(f"{'Algorithm':<10} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15} {'Total Points':<15}")
        print("-"*80)
        
        for result in problem_results:
            print(f"{result['algorithm']:<10} {result['runtime']:<15.2f} {result['hypervolume']:<15.6f} "
                  f"{result['pareto_size']:<15} {result.get('total_points', 0):<15}")
            
        # Add results to overall results
        all_results.extend(problem_results)
        
        # Plot comparisons for this problem
        if output_dir:
            problem_dir = os.path.join(output_dir, problem)
            os.makedirs(problem_dir, exist_ok=True)
            
            # Plot hypervolume convergence for all algorithms
            plt.figure(figsize=(12, 8))
            for result in problem_results:
                if 'hypervolume_history' in result and result['hypervolume_history']:
                    plt.plot(
                        range(1, len(result['hypervolume_history']) + 1), 
                        result['hypervolume_history'],
                        label=result['algorithm'],
                        marker='o',
                        markersize=4,
                        linewidth=2
                    )
                    
            plt.title(f'Hypervolume Convergence on {problem}')
            plt.xlabel('Iteration')
            plt.ylabel('Hypervolume')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(problem_dir, f'hypervolume_convergence_{problem}.png'))
            plt.close()
            
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
                plt.figure(figsize=(12, 8))
                
                # Dictionary to store point counts
                point_counts = {}
                
                for result in problem_results:
                    pareto_y = np.array(result['pareto_y'])
                    all_y = np.array(result['all_evaluated_y'])
                    algo_name = result['algorithm']
                    point_counts[algo_name] = {
                        'pareto': len(pareto_y),
                        'total': len(all_y)
                    }
                    
                    # Plot all evaluated points with low opacity
                    if len(all_y) > 0:
                        plt.scatter(all_y[:, 0], all_y[:, 1], alpha=0.3, s=20, 
                                   label=f"{algo_name} all points ({len(all_y)})")
                    
                    # Plot Pareto front with higher opacity
                    if len(pareto_y) > 0:
                        plt.scatter(pareto_y[:, 0], pareto_y[:, 1], alpha=0.8, s=80, 
                                  label=f"{algo_name} Pareto ({len(pareto_y)})")
                        
                        if len(pareto_y) > 1:
                            sorted_pareto = pareto_y[pareto_y[:, 0].argsort()]
                            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], '--')
                
                # Add a summary of point counts in the top-left corner
                summary_text = "\n".join([f"{algo}: {counts['total']} total, {counts['pareto']} Pareto" 
                                        for algo, counts in point_counts.items()])
                plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
                        fontsize=10, fontweight='bold', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
                
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
        metrics = ['runtime', 'hypervolume', 'pareto_size', 'total_points']
        
        for result in all_results:
            algo = result['algorithm']
            if algo not in data_by_algo:
                data_by_algo[algo] = {
                    'problems': [], 
                    'runtime': [], 
                    'hypervolume': [], 
                    'pareto_size': [],
                    'total_points': [],
                    'hypervolume_history': []
                }
            
            data_by_algo[algo]['problems'].append(result['problem'])
            data_by_algo[algo]['runtime'].append(result['runtime'])
            data_by_algo[algo]['hypervolume'].append(result['hypervolume'])
            data_by_algo[algo]['pareto_size'].append(result['pareto_size'])
            data_by_algo[algo]['total_points'].append(result.get('total_points', 0))
            if 'hypervolume_history' in result:
                data_by_algo[algo]['hypervolume_history'].append(result['hypervolume_history'])
        
        # Plot average hypervolume convergence across problems
        plt.figure(figsize=(12, 8))
        for algo in algorithms:
            if algo in data_by_algo and 'hypervolume_history' in data_by_algo[algo] and data_by_algo[algo]['hypervolume_history']:
                # Find the maximum length of hypervolume history
                histories = data_by_algo[algo]['hypervolume_history']
                if not histories:
                    continue
                    
                max_len = max([len(h) for h in histories if h])
                if max_len == 0:
                    continue
                
                # Pad shorter histories with their last value
                padded_histories = []
                for h in histories:
                    if not h:
                        continue
                    history = h.copy()
                    if len(history) < max_len:
                        history.extend([history[-1]] * (max_len - len(history)))
                    padded_histories.append(history)
                
                if not padded_histories:
                    continue
                
                # Calculate average history
                avg_history = np.mean(padded_histories, axis=0)
                
                # Plot average history
                plt.plot(
                    range(1, len(avg_history) + 1),
                    avg_history,
                    label=algo,
                    marker='o',
                    markersize=4,
                    linewidth=2
                )
        
        plt.title('Average Hypervolume Convergence Across All Problems')
        plt.xlabel('Iteration')
        plt.ylabel('Average Hypervolume')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_hypervolume_convergence.png'))
        plt.close()
        
        # Create average runtime comparison
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
        
        # Average total points by algorithm 
        plt.figure(figsize=(12, 8))
        avg_points = [np.mean(data_by_algo[algo]['total_points']) for algo in algorithms]
        plt.bar(algorithms, avg_points)
        plt.title('Average Total Points Evaluated Across All Problems')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Points Evaluated')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, 'avg_points_comparison.png'))
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
                                elif metric == 'total_points' and 'total_points' in result:
                                    data[i, j] = result['total_points']
                                elif metric in result:
                                    # For other metrics, higher is better
                                    data[i, j] = result[metric]
                                break
                
                plt.imshow(data, cmap='viridis')
                plt.colorbar(label=metric.capitalize().replace('_', ' '))
                plt.xticks(np.arange(len(problems)), problems, rotation=45)
                plt.yticks(np.arange(len(algorithms)), algorithms)
                plt.title(f'{metric.capitalize().replace("_", " ")} Comparison')
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
            f.write(f"{'Algorithm':<10} {'Avg Runtime (s)':<15} {'Avg Hypervolume':<20} {'Avg Pareto Size':<15} {'Avg Total Points':<15}\n")
            
            for algo in algorithms:
                avg_runtime = np.mean(data_by_algo[algo]['runtime'])
                avg_hv = np.mean(data_by_algo[algo]['hypervolume'])
                avg_pareto_size = np.mean(data_by_algo[algo]['pareto_size'])
                avg_total_points = np.mean(data_by_algo[algo]['total_points'])
                
                f.write(f"{algo:<10} {avg_runtime:<15.2f} {avg_hv:<20.6f} {avg_pareto_size:<15.2f} {avg_total_points:<15.2f}\n")
            
            f.write("\n\n")
            
            # Individual results
            f.write("All Results\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Algorithm':<10} {'Problem':<15} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15} {'Total Points':<15}\n")
            
            for result in all_results:
                total_pts = result.get('total_points', 0)
                f.write(f"{result['algorithm']:<10} {result['problem']:<15} {result['runtime']:<15.2f} "
                        f"{result['hypervolume']:<15.6f} {result['pareto_size']:<15} {total_pts:<15}\n")

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
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    args.output_dir = args.output_dir + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

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
                output_dir=args.output_dir,
                verbose=args.verbose
            )
        else:
            # Compare algorithms on a single problem
            compare_algorithms(
                problem_name=args.problem,
                algorithms=args.algorithms,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                verbose=args.verbose
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
                    output_dir=os.path.join(args.output_dir, problem) if args.output_dir else None,
                    verbose=args.verbose
                )
        else:
            # Run a single algorithm on a single problem
            run_optimization(
                algorithm_name=args.algorithm,
                problem_name=args.problem,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                verbose=args.verbose
            )
    else:
        print("Please specify an algorithm with --algorithm or use --compare to compare algorithms")

if __name__ == '__main__':
    main()