import numpy as np
from typing import List, Tuple, Optional, Union, Any
import numpy.typing as npt

def get_pareto_front(objective_values: npt.NDArray) -> npt.NDArray:
    """
    Extract the Pareto front from a set of objective values.
    For minimization problems.
    
    :param objective_values: Array of shape (n_points, n_objectives)
    :return: Boolean mask of Pareto optimal points
    """
    is_efficient = np.ones(len(objective_values), dtype=bool)
    for i, c in enumerate(objective_values):
        if is_efficient[i]:
            # Keep any point with at least one objective better than (less than) others
            # This keeps all non-dominated points
            is_efficient[is_efficient] = np.any(objective_values[is_efficient] < c, axis=1) | (
                np.all(objective_values[is_efficient] == c, axis=1))
            is_efficient[i] = True  # Keep self in efficient set
    
    return objective_values[is_efficient]

def calculate_hypervolume(points: npt.NDArray, reference_point: npt.NDArray) -> float:
    """
    Calculate hypervolume indicator for a set of points.
    For minimization problems.
    
    :param points: Array of shape (n_points, n_objectives)
    :param reference_point: Reference point for hypervolume calculation
    :return: Hypervolume value
    """
    # For now, implement simple 2D hypervolume calculation
    if points.shape[1] == 2:
        # Sort points by first objective
        sorted_points = points[points[:, 0].argsort()]
        
        # Calculate hypervolume
        hv = 0.0
        prev_x = reference_point[0]
        
        for i, point in enumerate(sorted_points):
            # Skip points beyond reference point
            if point[0] >= reference_point[0] or point[1] >= reference_point[1]:
                continue
                
            # Calculate area of rectangle
            width = prev_x - point[0]
            if i == 0:
                height = reference_point[1] - point[1]
            else:
                # Use max to ensure we're not double-counting
                height = max(0, sorted_points[i-1][1] - point[1])
                
            hv += width * height
            prev_x = point[0]
        
        return hv
    else:
        # For more than 2 dimensions, we need a more sophisticated approach
        # Consider using PyGMO, pymoo, or another library for this
        raise NotImplementedError(
            "Hypervolume calculation for more than 2 dimensions is not implemented yet."
        ) 