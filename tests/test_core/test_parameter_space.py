import unittest
import json
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.parameter_space import ParameterSpace

class TestParameterSpace(unittest.TestCase):
    """Tests for the ParameterSpace class"""
    
    def test_continuous_parameter(self):
        """Test adding and retrieving continuous parameters"""
        space = ParameterSpace()
        space.add_continuous_parameter('learning_rate', 0.0001, 0.1, log_scale=True)
        
        self.assertIn('learning_rate', space.parameters)
        param = space.parameters['learning_rate']
        self.assertEqual(param['type'], 'continuous')
        self.assertEqual(param['bounds'], [0.0001, 0.1])
        self.assertEqual(param['log_scale'], True)
    
    def test_integer_parameter(self):
        """Test adding and retrieving integer parameters"""
        space = ParameterSpace()
        space.add_integer_parameter('batch_size', 1, 256)
        
        self.assertIn('batch_size', space.parameters)
        param = space.parameters['batch_size']
        self.assertEqual(param['type'], 'integer')
        self.assertEqual(param['bounds'], [1, 256])
    
    def test_categorical_parameter(self):
        """Test adding and retrieving categorical parameters"""
        space = ParameterSpace()
        categories = ['adam', 'sgd', 'rmsprop']
        space.add_categorical_parameter('optimizer', categories)
        
        self.assertIn('optimizer', space.parameters)
        param = space.parameters['optimizer']
        self.assertEqual(param['type'], 'categorical')
        self.assertEqual(param['categories'], categories)
    
    def test_to_dict(self):
        """Test converting parameter space to dictionary"""
        space = ParameterSpace()
        space.add_continuous_parameter('learning_rate', 0.0001, 0.1, log_scale=True)
        space.add_integer_parameter('batch_size', 1, 256)
        space.add_categorical_parameter('optimizer', ['adam', 'sgd', 'rmsprop'])
        
        space_dict = space.to_dict()
        self.assertIn('parameters', space_dict)
        self.assertEqual(len(space_dict['parameters']), 3)
        self.assertIn('learning_rate', space_dict['parameters'])
        self.assertIn('batch_size', space_dict['parameters'])
        self.assertIn('optimizer', space_dict['parameters'])
    
    def test_from_dict(self):
        """Test creating parameter space from dictionary"""
        config = {
            'parameters': {
                'learning_rate': {
                    'type': 'continuous',
                    'bounds': [0.0001, 0.1],
                    'log_scale': True
                },
                'batch_size': {
                    'type': 'integer',
                    'bounds': [1, 256]
                },
                'optimizer': {
                    'type': 'categorical',
                    'categories': ['adam', 'sgd', 'rmsprop']
                }
            }
        }
        
        space = ParameterSpace.from_dict(config)
        self.assertEqual(len(space.parameters), 3)
        self.assertIn('learning_rate', space.parameters)
        self.assertIn('batch_size', space.parameters)
        self.assertIn('optimizer', space.parameters)
        
        # Check specific parameter
        param = space.parameters['learning_rate']
        self.assertEqual(param['type'], 'continuous')
        self.assertEqual(param['bounds'], [0.0001, 0.1])
        self.assertEqual(param['log_scale'], True)
    
    def test_to_json(self):
        """Test converting parameter space to JSON"""
        space = ParameterSpace()
        space.add_continuous_parameter('learning_rate', 0.0001, 0.1, log_scale=True)
        
        json_str = space.to_json()
        # Parse back to ensure it's valid JSON
        parsed = json.loads(json_str)
        self.assertIn('parameters', parsed)
        self.assertIn('learning_rate', parsed['parameters'])
    
    def test_from_json(self):
        """Test creating parameter space from JSON"""
        json_str = '''
        {
            "parameters": {
                "learning_rate": {
                    "type": "continuous",
                    "bounds": [0.0001, 0.1],
                    "log_scale": true
                }
            }
        }
        '''
        
        space = ParameterSpace.from_json(json_str)
        self.assertEqual(len(space.parameters), 1)
        self.assertIn('learning_rate', space.parameters)
        
        param = space.parameters['learning_rate']
        self.assertEqual(param['type'], 'continuous')
        self.assertEqual(param['bounds'], [0.0001, 0.1])
        self.assertEqual(param['log_scale'], True)
    
    def test_method_chaining(self):
        """Test method chaining for parameter addition"""
        space = (ParameterSpace()
                .add_continuous_parameter('learning_rate', 0.0001, 0.1)
                .add_integer_parameter('batch_size', 1, 256)
                .add_categorical_parameter('optimizer', ['adam', 'sgd']))
        
        self.assertEqual(len(space.parameters), 3)
        self.assertIn('learning_rate', space.parameters)
        self.assertIn('batch_size', space.parameters)
        self.assertIn('optimizer', space.parameters)


if __name__ == '__main__':
    unittest.main() 