#!/usr/bin/env python3
import argparse
import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from typing import Dict, List, Any, Tuple, Optional, Union, Type

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test problems
from tests.test_algorithms.test_problems import list_test_problems

def run_individual_benchmark(algorithm, problem, budget, batch_size, output_dir):
    """Run a benchmark for a single algorithm on a single problem"""
    algorithm_dir = os.path.join(output_dir, problem, algorithm)
    os.makedirs(algorithm_dir, exist_ok=True)
    
    # Run the unified_benchmark for a single algorithm-problem pair
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'unified_benchmark.py'),
        '--algorithm', algorithm,
        '--problem', problem,
        '--budget', str(budget),
        '--batch-size', str(batch_size),
        '--output-dir', algorithm_dir
    ]
    
    print(f"Running benchmark: {algorithm} on {problem}")
    result_file = os.path.join(algorithm_dir, 'result.json')
    
    # Skip if already completed
    if os.path.exists(result_file):
        print(f"  Skipping {algorithm} on {problem} - already completed")
        with open(result_file, 'r') as f:
            return json.load(f)
    
    try:
        # Run the benchmark in a separate process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse the output to extract results
        runtime = None
        hypervolume = None
        pareto_size = None
        
        for line in stdout.split('\n'):
            if "Optimization completed in" in line:
                runtime = float(line.split('in')[1].split('seconds')[0].strip())
            elif "Final hypervolume:" in line:
                hypervolume = float(line.split(':')[1].strip())
            elif "Pareto front size:" in line:
                pareto_size = int(line.split(':')[1].strip())
        
        result = {
            'algorithm': algorithm,
            'problem': problem,
            'runtime': runtime,
            'hypervolume': hypervolume,
            'pareto_size': pareto_size,
            'success': True,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # Save the result to a file
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    except Exception as e:
        print(f"Error running {algorithm} on {problem}: {str(e)}")
        result = {
            'algorithm': algorithm,
            'problem': problem,
            'runtime': None,
            'hypervolume': None,
            'pareto_size': None,
            'success': False,
            'error': str(e)
        }
        
        # Save the result to a file
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result

def generate_comparison_plots(results, output_dir):
    """Generate comparison plots from collected results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by problem
    results_by_problem = {}
    for result in results:
        problem = result['problem']
        if problem not in results_by_problem:
            results_by_problem[problem] = []
        results_by_problem[problem].append(result)
    
    # Group results by algorithm
    results_by_algorithm = {}
    for result in results:
        algorithm = result['algorithm']
        if algorithm not in results_by_algorithm:
            results_by_algorithm[algorithm] = []
        results_by_algorithm[algorithm].append(result)
    
    # Plot comparisons for each problem
    for problem, problem_results in results_by_problem.items():
        problem_dir = os.path.join(output_dir, problem)
        os.makedirs(problem_dir, exist_ok=True)
        
        # Filter out failed results
        problem_results = [r for r in problem_results if r['success'] and r['runtime'] is not None]
        
        if not problem_results:
            continue
        
        # Sort by algorithm name
        problem_results.sort(key=lambda r: r['algorithm'])
        
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
        plt.tight_layout()
        plt.savefig(os.path.join(problem_dir, f'runtime_comparison.png'))
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
        plt.tight_layout()
        plt.savefig(os.path.join(problem_dir, f'hypervolume_comparison.png'))
        plt.close()
        
        # Bar chart for Pareto front sizes
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r['algorithm'] for r in problem_results],
            [r['pareto_size'] for r in problem_results]
        )
        plt.title(f'Pareto Front Size Comparison on {problem}')
        plt.xlabel('Algorithm')
        plt.ylabel('Pareto Front Size')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(problem_dir, f'pareto_size_comparison.png'))
        plt.close()
    
    # Create overall comparison plots
    # Average runtime by algorithm
    algorithms = list(results_by_algorithm.keys())
    
    # Average metrics
    avg_runtimes = []
    avg_hypervolumes = []
    avg_pareto_sizes = []
    
    for algorithm in algorithms:
        algo_results = [r for r in results_by_algorithm[algorithm] if r['success'] and r['runtime'] is not None]
        if algo_results:
            avg_runtimes.append(np.mean([r['runtime'] for r in algo_results]))
            avg_hypervolumes.append(np.mean([r['hypervolume'] for r in algo_results]))
            avg_pareto_sizes.append(np.mean([r['pareto_size'] for r in algo_results]))
        else:
            avg_runtimes.append(0)
            avg_hypervolumes.append(0)
            avg_pareto_sizes.append(0)
    
    # Average runtime comparison
    plt.figure(figsize=(12, 8))
    plt.bar(algorithms, avg_runtimes)
    plt.title('Average Runtime Across All Problems')
    plt.xlabel('Algorithm')
    plt.ylabel('Average Runtime (seconds)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_runtime_comparison.png'))
    plt.close()
    
    # Average hypervolume comparison
    plt.figure(figsize=(12, 8))
    plt.bar(algorithms, avg_hypervolumes)
    plt.title('Average Hypervolume Across All Problems')
    plt.xlabel('Algorithm')
    plt.ylabel('Average Hypervolume')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_hypervolume_comparison.png'))
    plt.close()
    
    # Average Pareto front size comparison
    plt.figure(figsize=(12, 8))
    plt.bar(algorithms, avg_pareto_sizes)
    plt.title('Average Pareto Front Size Across All Problems')
    plt.xlabel('Algorithm')
    plt.ylabel('Average Pareto Front Size')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_pareto_size_comparison.png'))
    plt.close()
    
    # Create heatmaps for each metric
    problems = list(results_by_problem.keys())
    metrics = ['runtime', 'hypervolume', 'pareto_size']
    
    if len(problems) > 1 and len(algorithms) > 1:
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            data = np.zeros((len(algorithms), len(problems)))
            
            for i, algorithm in enumerate(algorithms):
                for j, problem in enumerate(problems):
                    for result in results:
                        if (result['algorithm'] == algorithm and 
                            result['problem'] == problem and 
                            result['success'] and 
                            result[metric] is not None):
                            if metric == 'runtime':
                                # For runtime, lower is better so we negate
                                data[i, j] = -result[metric]
                            else:
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
        
        for i, algorithm in enumerate(algorithms):
            f.write(f"{algorithm:<10} {avg_runtimes[i]:<15.2f} {avg_hypervolumes[i]:<20.6f} {avg_pareto_sizes[i]:<15.2f}\n")
        
        f.write("\n\n")
        
        # Individual results
        f.write("All Results\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Algorithm':<10} {'Problem':<15} {'Runtime (s)':<15} {'Hypervolume':<15} {'Pareto Size':<15}\n")
        
        for result in results:
            if result['success'] and result['runtime'] is not None:
                f.write(f"{result['algorithm']:<10} {result['problem']:<15} {result['runtime']:<15.2f} "
                        f"{result['hypervolume']:<15.6f} {result['pareto_size']:<15}\n")
            else:
                f.write(f"{result['algorithm']:<10} {result['problem']:<15} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15}\n")

def main():
    parser = argparse.ArgumentParser(description='Batch benchmark for optimization algorithms')
    parser.add_argument('--problems', type=str, nargs='+', help='Problems to benchmark')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['qnehvi', 'qehvi', 'qparego', 'nsga2', 'moead', 'nsga3'], 
                        help='Algorithms to benchmark')
    parser.add_argument('--budget', type=int, default=20, help='Evaluation budget')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='output/batch_benchmark', help='Directory to save results')
    parser.add_argument('--list-problems', action='store_true', help='List available test problems')
    
    args = parser.parse_args()
    
    # List available problems if requested
    if args.list_problems:
        print("Available test problems:")
        for problem in list_test_problems():
            print(f"  {problem}")
        return
    
    # Use all available problems if none specified
    if not args.problems:
        args.problems = list_test_problems()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    all_results = []
    
    for problem in args.problems:
        for algorithm in args.algorithms:
            result = run_individual_benchmark(
                algorithm=algorithm,
                problem=problem,
                budget=args.budget,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
            all_results.append(result)
    
    # Generate comparison plots
    generate_comparison_plots(all_results, args.output_dir)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("="*80)
    print(f"{'Algorithm':<10} {'Problem':<12} {'Runtime (s)':<12} {'Hypervolume':<12} {'Pareto Size':<12}")
    print("-"*80)
    
    for result in all_results:
        if result['success'] and result['runtime'] is not None:
            print(f"{result['algorithm']:<10} {result['problem']:<12} {result['runtime']:<12.2f} "
                  f"{result['hypervolume']:<12.6f} {result['pareto_size']:<12}")
        else:
            print(f"{result['algorithm']:<10} {result['problem']:<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")

if __name__ == '__main__':
    main() 