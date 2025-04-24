import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.core.framework_adapter import FrameworkAdapter
from src.core.parameter_space import ParameterSpace

class BoTorchAdapter(FrameworkAdapter):
    """Adapter for BoTorch parameter space and optimization components"""
    
    def __init__(self, parameter_space: ParameterSpace):
        super().__init__(parameter_space)
        self.botorch_space = self.to_framework_format()
    
    def to_framework_format(self):
        """Convert generic parameter space to BoTorch tensor format"""
        # Create bounds and handle categorical parameters
        bounds = []
        continuous_dims = []
        categorical_maps = {}
        
        for i, (name, param_config) in enumerate(self.parameter_space.parameters.items()):
            if param_config['type'] in ['continuous', 'integer']:
                bounds.append(param_config['bounds'])
                continuous_dims.append(i)
            elif param_config['type'] == 'categorical':
                # Map categorical to integers
                categories = param_config['categories']
                categorical_maps[name] = {
                    'idx': i,
                    'map': {j: cat for j, cat in enumerate(categories)},
                    'reverse_map': {cat: j for j, cat in enumerate(categories)}
                }
                bounds.append([0, len(categories) - 1])
        
        return {
            'bounds': torch.tensor(bounds, dtype=torch.float),
            'categorical_maps': categorical_maps,
            'continuous_dims': continuous_dims
        }
    
    def from_framework_format(self, framework_params):
        """Convert BoTorch tensor to standard dictionary"""
        result = {}
        
        # Check if framework_params is a tensor or a dictionary
        if isinstance(framework_params, torch.Tensor):
            # Convert tensor to numpy array for processing
            params_array = framework_params.cpu().numpy()
            
            # Map each parameter back to its original space
            idx = 0
            for name, param_config in self.parameter_space.parameters.items():
                if param_config['type'] == 'continuous':
                    result[name] = float(params_array[idx])
                    idx += 1
                elif param_config['type'] == 'integer':
                    # Round to nearest integer
                    result[name] = int(round(float(params_array[idx])))
                    idx += 1
                elif param_config['type'] == 'categorical':
                    # Map integer back to category
                    cat_idx = int(round(float(params_array[idx])))
                    cat_map = self.botorch_space['categorical_maps'][name]['map']
                    result[name] = cat_map[cat_idx]
                    idx += 1
        else:
            # It's already a dictionary, just ensure types are correct
            for name, param_config in self.parameter_space.parameters.items():
                if name in framework_params:
                    value = framework_params[name]
                    if param_config['type'] == 'continuous':
                        result[name] = float(value)
                    elif param_config['type'] == 'integer':
                        result[name] = int(round(float(value)))
                    elif param_config['type'] == 'categorical':
                        result[name] = value
        
        return result
    
    def convert_results(self, framework_results):
        """Convert BoTorch optimization results to standard format"""
        # Extract Pareto front and solutions
        if 'pareto_front' in framework_results and 'pareto_points' in framework_results:
            pareto_front = framework_results['pareto_front']
            pareto_points = framework_results['pareto_points']
            
            # Convert to numpy arrays if they're tensors
            if isinstance(pareto_front, torch.Tensor):
                pareto_front = pareto_front.cpu().numpy()
            if isinstance(pareto_points, torch.Tensor):
                pareto_points = pareto_points.cpu().numpy()
            
            # Convert parameter vectors to dictionaries
            pareto_xs = [self.from_framework_format(point) for point in pareto_points]
            pareto_ys = pareto_front
            
            return pareto_xs, pareto_ys
        
        # Otherwise, return empty results
        return [], [] 