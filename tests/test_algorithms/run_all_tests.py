#!/usr/bin/env python3
import unittest
import sys
import os
import argparse
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_unit_tests():
    """Run all unit tests for the test problems"""
    print("\n=== Running Unit Tests ===\n")
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Discover tests in this directory
    test_suite = loader.discover(os.path.dirname(__file__), pattern='*_test.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_example_benchmark():
    """Run the example benchmark script"""
    print("\n=== Running Example Benchmark ===\n")
    
    # Path to example script
    example_script = os.path.join(os.path.dirname(__file__), 'example_benchmark.py')
    
    # Run the script
    try:
        subprocess.run([sys.executable, example_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Example benchmark failed!")
        return False


def run_benchmark_script(problem=None, optimizer=None, budget=None, batch_size=None):
    """Run the main benchmark script with specified parameters"""
    print("\n=== Running Benchmark Script ===\n")
    
    # Path to benchmark script
    benchmark_script = os.path.join(os.path.dirname(__file__), 'run_benchmark.py')
    
    # Prepare command
    cmd = [sys.executable, benchmark_script]
    
    if problem and optimizer:
        # Run specific benchmark
        cmd.extend(['--problem', problem, '--optimizer', optimizer])
        
        if budget:
            cmd.extend(['--budget', str(budget)])
        
        if batch_size:
            cmd.extend(['--batch-size', str(batch_size)])
    elif problem:
        # Compare all optimizers
        cmd.extend(['--problem', problem, '--compare'])
        
        if budget:
            cmd.extend(['--budget', str(budget)])
        
        if batch_size:
            cmd.extend(['--batch-size', str(batch_size)])
    else:
        # Just list available problems and optimizers
        cmd.extend(['--list-problems'])
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("Failed to list problems!")
            return False
        
        cmd = [sys.executable, benchmark_script, '--list-optimizers']
    
    # Run the script
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Benchmark script failed!")
        return False


def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description='Run all algorithm tests and benchmarks')
    parser.add_argument('--unit-tests', action='store_true', help='Run unit tests only')
    parser.add_argument('--example', action='store_true', help='Run example benchmark only')
    parser.add_argument('--benchmark', action='store_true', help='Run specific benchmark')
    parser.add_argument('--problem', type=str, help='Test problem to optimize')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use')
    parser.add_argument('--budget', type=int, help='Evaluation budget')
    parser.add_argument('--batch-size', type=int, help='Batch size for evaluations')
    
    args = parser.parse_args()
    
    if args.unit_tests:
        success = run_unit_tests()
    elif args.example:
        success = run_example_benchmark()
    elif args.benchmark:
        success = run_benchmark_script(args.problem, args.optimizer, args.budget, args.batch_size)
    else:
        # Run all tests by default
        success1 = run_unit_tests()
        success2 = run_example_benchmark()
        success = success1 and success2
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 