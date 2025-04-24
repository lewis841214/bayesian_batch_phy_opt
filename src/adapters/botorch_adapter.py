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
    
    def from_framework_format(self, params_array):
        """Convert tensor params to a dict of parameter values"""
        if isinstance(params_array, torch.Tensor):
            # Ensure we're working with a 1D tensor
            params_array = params_array.squeeze()
            if params_array.dim() > 1:
                # If still multidimensional after squeeze, it's a batch
                # Just take the first element for now
                params_array = params_array[0]

        params_dict = {}
        idx = 0
        
        # Extract parameter values from ordered tensor
        for name, config in self.parameter_space.parameters.items():
            if config['type'] in ['continuous', 'integer']:
                # Continuous and integer parameters map directly
                val = float(params_array[idx])
                if config['type'] == 'integer':
                    val = int(round(val))
                params_dict[name] = val
                idx += 1
            elif config['type'] == 'categorical':
                # Categorical parameters need to be mapped back to categorical values
                try:
                    # Handle scalar values
                    if isinstance(params_array[idx], torch.Tensor):
                        cat_idx = int(round(float(params_array[idx].item())))
                    else:
                        cat_idx = int(round(float(params_array[idx])))
                    
                    # Ensure cat_idx is in valid range
                    cat_idx = max(0, min(cat_idx, len(config['categories']) - 1))
                    params_dict[name] = config['categories'][cat_idx]
                except Exception as e:
                    # Fallback to the first option if there's an error
                    print(f"Error mapping categorical value for {name}: {e}")
                    params_dict[name] = config['categories'][0]
                idx += 1
                
        return params_dict
    
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