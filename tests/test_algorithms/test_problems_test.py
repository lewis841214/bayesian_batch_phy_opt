import unittest
import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.test_algorithms.test_problems import (
    TestProblem, MixedParameterTestProblem, NonlinearTestProblem,
    DiscreteTestProblem, ConstrainedTestProblem, 
    get_test_problem, list_test_problems
)


class TestProblemsTestCase(unittest.TestCase):
    """Tests for the test problem implementations"""
    
    def test_mixed_parameter_problem(self):
        """Test the mixed parameter problem"""
        problem = MixedParameterTestProblem()
        
        # Test parameter space
        space = problem.get_parameter_space()
        self.assertEqual(len(space.parameters), 4)
        self.assertIn('x1', space.parameters)
        self.assertIn('x2', space.parameters)
        self.assertIn('x3', space.parameters)
        self.assertIn('x4', space.parameters)
        
        # Test evaluation
        params = {
            'x1': 0.0,
            'x2': 0.0,
            'x3': 1,
            'x4': 'option_a'
        }
        
        result = problem.evaluate(params)
        self.assertEqual(len(result), problem.num_objectives)
        self.assertEqual(result, [1.0, 3.0])  # f1 = 0^2 + 0^2 + 1 + 0, f2 = 1^2 + 1^2 + 1 + 0
        
        # Test with different parameter values
        params = {
            'x1': 1.0,
            'x2': 1.0,
            'x3': 5,
            'x4': 'option_b'
        }
        
        result = problem.evaluate(params)
        self.assertEqual(result, [7.5, 5.5])  # f1 = 1^2 + 1^2 + 5 + 0.5, f2 = 0^2 + 0^2 + 5 + 0.5
    
    def test_nonlinear_problem(self):
        """Test the nonlinear problem"""
        problem = NonlinearTestProblem()
        
        # Test parameter space
        space = problem.get_parameter_space()
        self.assertEqual(len(space.parameters), 3)
        self.assertIn('x1', space.parameters)
        self.assertIn('x2', space.parameters)
        self.assertIn('x3', space.parameters)
        
        # Test evaluation
        params = {
            'x1': 0.0,
            'x2': 0.0,
            'x3': 0.0
        }
        
        result = problem.evaluate(params)
        self.assertEqual(len(result), problem.num_objectives)
        self.assertEqual(result, [0.0, 5.0, 0.0])  # f1=0, f2=(0-2)^2+(0)^2+(0-1)^2=5, f3=0+0+0+0*0*0=0
        
        # Test with interaction term
        params = {
            'x1': 1.0,
            'x2': 2.0,
            'x3': 3.0
        }
        
        result = problem.evaluate(params)
        self.assertEqual(result[0], 14.0)  # f1 = 1^2 + 2^2 + 3^2 = 14
        self.assertEqual(result[1], 5.0)   # f2 = (1-2)^2 + 2^2 + (3-1)^2 = 1 + 4 + 4 = 9
        self.assertEqual(result[2], 12.0)  # f3 = 1 + 2 + 3 + 1*2*3 = 6 + 6 = 12
    
    def test_discrete_problem(self):
        """Test the discrete problem"""
        problem = DiscreteTestProblem()
        
        # Test parameter space
        space = problem.get_parameter_space()
        self.assertEqual(len(space.parameters), 5)
        self.assertIn('x1', space.parameters)
        self.assertIn('x2', space.parameters)
        self.assertIn('x3', space.parameters)
        self.assertIn('x4', space.parameters)
        self.assertIn('x5', space.parameters)
        
        # Test evaluation
        params = {
            'x1': 0.0,
            'x2': 1,
            'x3': 1,
            'x4': 'low',
            'x5': 'red'
        }
        
        result = problem.evaluate(params)
        self.assertEqual(len(result), problem.num_objectives)
        self.assertEqual(result, [2.0, 20.0])  # f1=0^2+1+1+0+0=2, f2=(1-0)^2+10-1+10-1+2-(0+0)=20
        
        # Test with different categorical values
        params = {
            'x1': 0.5,
            'x2': 5,
            'x3': 5,
            'x4': 'high',
            'x5': 'blue'
        }
        
        result = problem.evaluate(params)
        # f1 = 0.5^2 + 5 + 5 + (2.0 + 1.0) = 0.25 + 10 + 3 = 13.25
        # f2 = (1-0.5)^2 + (10-5) + (10-5) + (2-(2.0+1.0)) = 0.25 + 5 + 5 + (-1) = 9.25
        self.assertAlmostEqual(result[0], 13.25)
        self.assertAlmostEqual(result[1], 9.25)
    
    def test_constrained_problem(self):
        """Test the constrained problem"""
        problem = ConstrainedTestProblem()
        
        # Test parameter space
        space = problem.get_parameter_space()
        self.assertEqual(len(space.parameters), 2)
        self.assertIn('x1', space.parameters)
        self.assertIn('x2', space.parameters)
        
        # Verify lower bound for x1 enforces constraint
        self.assertEqual(space.parameters['x1']['bounds'][0], 0.1)
        
        # Test evaluation
        params = {
            'x1': 1.0,
            'x2': 0.0
        }
        
        result = problem.evaluate(params)
        self.assertEqual(len(result), problem.num_objectives)
        self.assertEqual(result, [1.0, 1.0])  # f1 = 1.0, f2 = (1+0)/1 = 1.0
        
        # Test with different values
        params = {
            'x1': 2.0,
            'x2': 3.0
        }
        
        result = problem.evaluate(params)
        self.assertEqual(result, [2.0, 2.0])  # f1 = 2.0, f2 = (1+3)/2 = 2.0
    
    def test_get_test_problem(self):
        """Test getting problems by name"""
        problems = list_test_problems()
        self.assertTrue(len(problems) > 0)
        
        for name in problems:
            problem = get_test_problem(name)
            self.assertIsInstance(problem, TestProblem)
            
        # Test with invalid name
        with self.assertRaises(ValueError):
            get_test_problem('invalid_name')


if __name__ == '__main__':
    unittest.main() 