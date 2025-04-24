from typing import Dict, List, Union, Optional, Any
import json

class ParameterSpace:
    """Framework-agnostic parameter space definition"""
    
    def __init__(self):
        self.parameters = {}
    
    def add_continuous_parameter(self, name: str, lower_bound: float, upper_bound: float, 
                                log_scale: bool = False):
        """Add a continuous parameter with bounds"""
        self.parameters[name] = {
            'type': 'continuous',
            'bounds': [lower_bound, upper_bound],
            'log_scale': log_scale
        }
        return self
    
    def add_integer_parameter(self, name: str, lower_bound: int, upper_bound: int):
        """Add an integer parameter with bounds"""
        self.parameters[name] = {
            'type': 'integer',
            'bounds': [lower_bound, upper_bound]
        }
        return self
    
    def add_categorical_parameter(self, name: str, categories: List[Any]):
        """Add a categorical parameter with possible values"""
        self.parameters[name] = {
            'type': 'categorical',
            'categories': categories
        }
        return self
    
    def to_dict(self) -> Dict:
        """Convert parameter space to dictionary"""
        return {
            'parameters': self.parameters
        }
    
    def to_json(self) -> str:
        """Convert parameter space to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'ParameterSpace':
        """Create parameter space from dictionary"""
        space = cls()
        space.parameters = config['parameters']
        return space
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ParameterSpace':
        """Create parameter space from JSON string"""
        return cls.from_dict(json.loads(json_str)) 