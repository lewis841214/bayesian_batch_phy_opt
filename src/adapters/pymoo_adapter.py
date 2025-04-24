import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.core.framework_adapter import FrameworkAdapter
from src.core.parameter_space import ParameterSpace

class PymooAdapter(FrameworkAdapter):
    """Adapter for Pymoo parameter space and optimization components"""
    
    def __init__(self, parameter_space: ParameterSpace):
        super().__init__(parameter_space)
        self.pymoo_space = self.to_framework_format()
    
    def to_framework_format(self):
        """Convert generic parameter space to Pymoo problem format"""
        # Define variable types and bounds for Pymoo
        var_types = []  # 'real', 'int' or 'cat'
        bounds = []  # Lower and upper bounds for each variable
        names = []  # Variable names
        categorical_maps = {}  # Maps for categorical variables
        
        for name, param_config in self.parameter_space.parameters.items():
            names.append(name)
            
            if param_config['type'] == 'continuous':
                var_types.append('real')
                bounds.append(param_config['bounds'])
            elif param_config['type'] == 'integer':
                var_types.append('int')
                bounds.append(param_config['bounds'])
            elif param_config['type'] == 'categorical':
                var_types.append('cat')
                categories = param_config['categories']
                # For categorical, bounds are [0, n_categories-1]
                bounds.append([0, len(categories) - 1])
                # Create mappings
                categorical_maps[name] = {
                    'map': {i: cat for i, cat in enumerate(categories)},
                    'reverse_map': {cat: i for i, cat in enumerate(categories)}
                }
        
        return {
            'var_types': var_types,
            'bounds': np.array(bounds),
            'names': names,
            'categorical_maps': categorical_maps
        }
    
    def from_framework_format(self, framework_params):
        """Convert Pymoo parameters to standard dictionary"""
        result = {}
        
        # Check if framework_params is a numpy array or a dictionary
        if isinstance(framework_params, np.ndarray):
            # Map each parameter back to its original space
            for i, name in enumerate(self.pymoo_space['names']):
                param_config = self.parameter_space.parameters[name]
                
                if param_config['type'] == 'continuous':
                    result[name] = float(framework_params[i])
                elif param_config['type'] == 'integer':
                    result[name] = int(framework_params[i])
                elif param_config['type'] == 'categorical':
                    # Map integer back to category
                    cat_idx = int(framework_params[i])
                    cat_map = self.pymoo_space['categorical_maps'][name]['map']
                    result[name] = cat_map[cat_idx]
        else:
            # It's already a dictionary, just ensure types are correct
            for name, param_config in self.parameter_space.parameters.items():
                if name in framework_params:
                    value = framework_params[name]
                    if param_config['type'] == 'continuous':
                        result[name] = float(value)
                    elif param_config['type'] == 'integer':
                        result[name] = int(value)
                    elif param_config['type'] == 'categorical':
                        result[name] = value
        
        return result
    
    def convert_results(self, framework_results):
        """Convert Pymoo optimization results to standard format"""
        # Extract Pareto front and solutions
        if hasattr(framework_results, 'F') and hasattr(framework_results, 'X'):
            pareto_front = framework_results.F
            pareto_points = framework_results.X
            
            # Convert parameter vectors to dictionaries
            pareto_xs = []
            for point in pareto_points:
                pareto_xs.append(self.from_framework_format(point))
            
            return pareto_xs, pareto_front
        
        # Otherwise, return empty results
        return [], [] 