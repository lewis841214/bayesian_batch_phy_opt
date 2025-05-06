"""
Variance calculator for the Flow Optimization Statistics project.

This module provides methods to calculate variance between multiple flow vectors.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import scipy.stats


def _mmd_rbf_1d(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Calculate the Maximum Mean Discrepancy (MMD) between two distributions.
    
    Treats 2D grids as 1D distributions by flattening them.
    Uses a Gaussian RBF kernel k(x,y) = exp(-gamma * ||x-y||^2).
    
    Args:
        x: First sample, can be 2D grid or other format
        y: Second sample, can be 2D grid or other format
        gamma: Parameter of the RBF kernel, controls the width
        
    Returns:
        The MMD distance between the two distributions
    """
    # Flatten 2D grids to 1D distributions
    if len(x.shape) == 2:
        x_flat = x.ravel()
        x_flat = x_flat / np.sum(x_flat)  # Normalize
    else:
        x_flat = x
        
    if len(y.shape) == 2:
        y_flat = y.ravel()
        y_flat = y_flat / np.sum(y_flat)  # Normalize
    else:
        y_flat = y
    
    # Create feature vectors with position information
    x_features = np.column_stack((np.arange(len(x_flat)), x_flat))
    y_features = np.column_stack((np.arange(len(y_flat)), y_flat))
    
    # Calculate kernel matrices
    xx_kernel = np.exp(-gamma * np.sum((x_features[:, None, :] - x_features[None, :, :]) ** 2, axis=2))
    xy_kernel = np.exp(-gamma * np.sum((x_features[:, None, :] - y_features[None, :, :]) ** 2, axis=2))
    yy_kernel = np.exp(-gamma * np.sum((y_features[:, None, :] - y_features[None, :, :]) ** 2, axis=2))
    
    # Calculate weighted MMD
    mmd = np.sum(x_flat[:, None] * x_flat[None, :] * xx_kernel)
    mmd += np.sum(y_flat[:, None] * y_flat[None, :] * yy_kernel)
    mmd -= 2 * np.sum(x_flat[:, None] * y_flat[None, :] * xy_kernel)
    
    return float(np.sqrt(max(0, mmd)))  

def _mmd_rbf( x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Calculate the Maximum Mean Discrepancy (MMD) between two distributions.
    
    Uses a Gaussian RBF kernel k(x,y) = exp(-gamma * ||x-y||^2).
    
    Args:
        x: First sample, shape (n_samples_x, n_features)
        y: Second sample, shape (n_samples_y, n_features)
        gamma: Parameter of the RBF kernel, controls the width
        
    Returns:
        The MMD distance between the two distributions
    """
    # Flatten 2D grids to sample points if needed
    if len(x.shape) == 2 and len(y.shape) == 2:
        nx, mx = x.shape
        ny, my = y.shape
        
        # Get grid coordinates
        x_coords = np.linspace(0, 1, nx)
        y_coords = np.linspace(0, 1, mx)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Convert to sample points with weights
        x_points = np.column_stack([xx.ravel(), yy.ravel()])
        y_points = np.column_stack([xx.ravel(), yy.ravel()])  # Same grid coordinates
        
        # Use distribution values as weights
        x_weights = x.ravel() / np.sum(x.ravel())  # Normalize
        y_weights = y.ravel() / np.sum(y.ravel())  # Normalize
        
        # Weighted MMD calculation
        xx_kernel = np.exp(-gamma * np.sum((x_points[:, None, :] - x_points[None, :, :]) ** 2, axis=2))
        xy_kernel = np.exp(-gamma * np.sum((x_points[:, None, :] - y_points[None, :, :]) ** 2, axis=2))
        yy_kernel = np.exp(-gamma * np.sum((y_points[:, None, :] - y_points[None, :, :]) ** 2, axis=2))
        
        # Weighted sums for MMD
        mmd = np.sum(np.outer(x_weights, x_weights) * xx_kernel) 
        mmd += np.sum(np.outer(y_weights, y_weights) * yy_kernel)
        mmd -= 2 * np.sum(np.outer(x_weights, y_weights) * xy_kernel)
        
        return float(np.sqrt(max(0, mmd)))  # Return square root to make it a proper metric
    
    # For 1D distributions or other formats, use standard MMD
    x_kernel = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    y_kernel = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2))
    xy_kernel = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
    
    mmd = np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)
    return float(np.sqrt(max(0, mmd)))  #

class VarianceCalculator:
    """Calculator for variance between multiple flow vectors."""
    
    def __init__(self):
        """Initialize the variance calculator."""
        # Memory dictionary to store calculated distances
        # Key format: (hash of distribution i, hash of distribution j, distance_function.__name__)
        self.memory_dict = {}
    
    def pairwise_distance_matrix(self, 
                               list_of_distributions: List[np.ndarray], 
                               distance_function: Callable[[np.ndarray, np.ndarray], float],
                               ) -> Dict[str, Any]:
        """Calculate the pairwise distance matrix between vectors.
        
        Args:
            list_of_distributions: List of arrays of shape (n_samples, n_features) containing the vectors.
                The arrays may have different lengths.
            distance_function: Callable[[np.ndarray, np.ndarray], float] that takes two vectors and returns a distance.
            
        Returns:
            Dictionary containing the distance matrix and summary statistics.
        """
        
        # Check if input is a list
    
        # Handle lists with potentially different-sized vectors
        n = len(list_of_distributions)
        # Initialize empty distance matrix
        distance_matrix = np.zeros((n, n))
        
        # Calculate pairwise distances manually
        distance_values = []
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle, automatically skips i == j cases
                # Simply convert to numpy array and create hash
                dist1_hash = hash(np.array(list_of_distributions[i]).tobytes())
                dist2_hash = hash(np.array(list_of_distributions[j]).tobytes())
                
                # Create a consistent key by always sorting the hashes
                cache_key = tuple(sorted([dist1_hash, dist2_hash]) + [distance_function.__name__])
                
                # Check if the calculation has been done before
                if cache_key in self.memory_dict:
                    dist = self.memory_dict[cache_key]
                else:
                    dist = distance_function(list_of_distributions[i], list_of_distributions[j])
                    self.memory_dict[cache_key] = dist
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Matrix is symmetric
                distance_values.append(dist)
        
        # Convert to numpy array for calculations
        distances = np.array(distance_values)
        
        # Note: The distances array only contains non-diagonal elements (i != j)
        # which is correct since diagonal elements (i == j) would be 0 and bias the result
        
        # For a matrix of size n×n, there are n(n-1)/2 unique pairs (excluding diagonal)
        # Our distances array contains exactly these values
        result = {
            'distance_matrix': distance_matrix, # .tolist(),
            'distance_function': distance_function.__name__,
            'mean_distance': float(np.sum(distances) / (n * (n - 1))) if n > 1 else 0.0,
            'raw_distances': distances.tolist() if len(distances) > 0 else []
        }
        
        return result
    
    def calculate_pairwise_1d_distribution_distance(self, 
                                        list_of_vectors: List[np.ndarray]) -> float:
        """Calculate the mean pairwise distance between vectors.
        
        Args:
            vectors: Array of shape (n_samples, n_features) containing the vectors.
            Each vector is a 1D distribution.

            Returns:
            Mean pairwise distance between vectors.

        Algorithm:
        - For each vector, we transform it into a given bin size distribution.
        - Then we calculate the 1d Wasserstein distance between each vector and every other vector. 
        - We return the mean of these distances.
        """

        
        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(list_of_vectors, distance_function= scipy.stats.wasserstein_distance)
        
        # Calculate the mean pairwise distance
        return distance_matrix['mean_distance']

    def _transform_grid_to_standard_size(self, grid: np.ndarray, target_size: int = 200) -> np.ndarray:
        """Transform a grid of any size into a standardized n×n grid.
        
        Args:
            grid: 2D array of any shape containing the probability distribution.
            target_size: Size of the target grid (default: 32).
            
        Returns:
            Array of shape (target_size, target_size) containing the resampled distribution.
            
        Note:
            Uses scipy's zoom function to resize the grid while preserving the total probability mass.
            The zoom factor is calculated to match the target size.
        """
        # Get current dimensions
        current_h, current_w = grid.shape
        
        # Calculate zoom factors
        zoom_h = target_size / current_h
        zoom_w = target_size / current_w
        
        # Resize the grid
        resized_grid = zoom(grid, (zoom_h, zoom_w), order=1)
        
        # Normalize to ensure probability distribution
        resized_grid = resized_grid / np.sum(resized_grid)
        
        return resized_grid

    def _2d_wasserstein_distance_sinkhorn(self, grid1: np.ndarray, grid2: np.ndarray, 
                                        reg: float = 0.01, num_iter: int = 1000,
                                        target_size: int = 32) -> float:
        """Calculate the 2D Wasserstein distance using Sinkhorn algorithm.
        
        Args:
            grid1: First 2D grid of weights (distribution)
            grid2: Second 2D grid of weights (distribution)
            reg: Regularization parameter (smaller values are more accurate but less stable)
            num_iter: Number of iterations for the Sinkhorn algorithm
            target_size: Size of the target grid (not used in this implementation)
            
        Returns:
            Approximated 2D Wasserstein distance
        """
        # Get grid dimensions
        n, m = grid1.shape
        
        # Create cost matrix using the proper [0,1]×[0,1] domain
        x_coords = np.linspace(0, 1, n)  # Already in [0,1] range
        y_coords = np.linspace(0, 1, m)  # Already in [0,1] range
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten coordinates
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Build cost matrix more efficiently using vectorized operations
        cost_matrix = np.zeros((n*m, n*m))
        for i in range(n*m):
            # Calculate Euclidean distances from point i to all other points
            diff = positions - positions[i]
            cost_matrix[i, :] = np.sqrt(np.sum(diff**2, axis=1))
        
        # Flatten grid distributions and add small epsilon to avoid zeros
        epsilon = 1e-10
        a = grid1.ravel() + epsilon
        b = grid2.ravel() + epsilon
        
        # Ensure mass conservation
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        # Stabilized Sinkhorn iterations with log-space calculations
        K = np.exp(-cost_matrix / reg)
        
        # Initialize in log space to improve numerical stability
        u = np.zeros(n*m)
        v = np.zeros(n*m)
        
        # Sinkhorn iterations with better numerical stability
        for _ in range(num_iter):
            # Update u (in log space with numerical stabilization)
            Kv = K.dot(np.exp(v / reg))
            u = reg * np.log(a) - reg * np.log(np.maximum(Kv, 1e-16))
            
            # Update v (in log space with numerical stabilization)
            Ku = np.exp(u / reg).dot(K)
            v = reg * np.log(b) - reg * np.log(np.maximum(Ku, 1e-16))
        
        # Compute optimal transport plan
        P = np.exp((u.reshape(-1, 1) + v.reshape(1, -1) - cost_matrix) / reg)
        
        # Compute Wasserstein distance
        distance = np.sum(P * cost_matrix)
        
        return float(distance)

    def _2d_wasserstein_distance_sliced(self, grid1: np.ndarray, grid2: np.ndarray,
                                      num_projections: int = 100) -> float:
        """Calculate the 2D Wasserstein distance using sliced Wasserstein distance.
        
        This implementation projects both distributions onto random lines and computes
        the 1D Wasserstein distance along each projection. The final distance is the
        average of these 1D distances, scaled by √2 to account for the [0,1]×[0,1] domain.
        
        Args:
            grid1: First 2D grid of weights (distribution)
            grid2: Second 2D grid of weights (distribution)
            num_projections: Number of random projections to use
            
        Returns:
            Approximation of the 2D Wasserstein distance
        """
        # Get grid dimensions
        n, m = grid1.shape
        
        # Pre-compute coordinates once
        x_coords = np.linspace(0, 1, n)  # Already in [0,1] range
        y_coords = np.linspace(0, 1, m)  # Already in [0,1] range
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Use deterministic angles for better coverage
        angles = np.linspace(0, 2*np.pi, num_projections, endpoint=False)
        
        total_distance = 0.0
        
        for angle in angles:
            # Create projection vector
            projection = np.array([np.cos(angle), np.sin(angle)])
            
            # Project the coordinates onto the projection vector
            # We use the same coordinates for both distributions because they represent
            # the same spatial locations, but with different weights
            proj_coords = xx.ravel() * projection[0] + yy.ravel() * projection[1]
            
            # Compute 1D Wasserstein distance directly using scipy
            # Use the SAME projected coordinates for both distributions since they represent the same points,
            # but with different weights from each grid
            total_distance += scipy.stats.wasserstein_distance(
                proj_coords, proj_coords,  # Same projection coordinates for both 
                u_weights=grid1.ravel(),
                v_weights=grid2.ravel()
            )
        
        # Average over all projections and scale by domain size
        # The factor of √2 accounts for the maximum distance in the [0,1]×[0,1] domain
        return float(total_distance / num_projections) * np.sqrt(2)

    def _2d_wasserstein_distance(self, grid1: np.ndarray, grid2: np.ndarray, 
                               method: str = 'sinkhorn',
                               reg: float = 0.01,
                               num_iter: int = 1000,
                               num_projections: int = 100,
                               target_size: int = 32,
                               gamma: float = 1.0) -> float:
        """Calculate the 2D Wasserstein distance using specified method.
        
        Args:
            grid1: 2D array containing the first probability distribution.
            grid2: 2D array containing the second probability distribution.
            method: 'sinkhorn' or 'sliced' (default: 'sinkhorn').
            reg: Regularization parameter for Sinkhorn algorithm.
            num_iter: Number of iterations for Sinkhorn algorithm.
            num_projections: Number of projections for sliced method.
            target_size: Size of the standardized grid.
            
        Returns:
            float: The 2D Wasserstein distance between the two distributions.
        """
        if method == 'sinkhorn':
            return self._2d_wasserstein_distance_sinkhorn(
                grid1, grid2, reg, num_iter, target_size
            )
        elif method == 'sliced':
            return self._2d_wasserstein_distance_sliced(
                grid1, grid2, num_projections
            )
        elif method == 'mmd':
            return _mmd_rbf(
                grid1, grid2, gamma
            )
        elif method == 'mmd_1d':
            return _mmd_rbf_1d(
                grid1, grid2, gamma
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sinkhorn' or 'sliced'.")

    def calculate_pairwise_2d_distribution_distance(self, 
                                        list_of_grids: List[np.ndarray],
                                        method: str = 'sliced',
                                        reg: float = 0.1,
                                        num_iter: int = 100,
                                        num_projections: int = 100,
                                        target_size: int = 200,
                                        gamma: float = 1.0
                                        ) -> float:
        """Calculate the mean pairwise distance between 2D grid distributions.
        
        Args:
            list_of_grids: List of 2D arrays, each representing a grid distribution.
            reg: Regularization parameter for Sinkhorn algorithm.
            num_iter: Number of iterations for Sinkhorn algorithm.
            target_size: Size of the standardized grid.
            
        Returns:
            float: Mean pairwise distance between all grid distributions.
        """
        if target_size != None:
            # Transform each grid into a standardized size
            list_of_grids = [self._transform_grid_to_standard_size(grid, target_size) for grid in list_of_grids]   

        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(
            list_of_grids, 
            distance_function=lambda x, y: self._2d_wasserstein_distance(
                x, y, method, reg, num_iter, num_projections, target_size, gamma
            )
        )
        
        # Return mean distance
        return distance_matrix# ['mean_distance']
    
    def calculate_pairwise_scatter_distribution_distance(self, 
                                        list_of_distributions: List[np.ndarray]) -> float:
        """Calculate the mean pairwise distance between scatter distributions.
        
        Args:
            list_of_distributions: List of 2D arrays, each representing a scatter distribution.
        
        Returns:
            float: Mean pairwise distance between all scatter distributions.
        """
        
        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(
            list_of_distributions, 
            distance_function = scipy.stats.wasserstein_distance_nd
        )
        
        # Return mean distance
        return distance_matrix['mean_distance']

    def collect_pairwise_1d_distribution_distances(self, list_of_vectors: List[np.ndarray]) -> List[float]:
        """Collect all pairwise distances between 1D distribution vectors without averaging.
        
        Args:
            list_of_vectors: List of arrays containing the 1D distributions.
            
        Returns:
            List of all pairwise distances between vectors.
        """
        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(list_of_vectors, distance_function=scipy.stats.wasserstein_distance)
        
        # Return the raw distances
        return distance_matrix['raw_distances']
        
    def collect_pairwise_2d_distribution_distances(self, 
                                        list_of_grids: List[np.ndarray],
                                        method: str = 'sliced',
                                        reg: float = 0.1,
                                        num_iter: int = 100,
                                        num_projections: int = 100,
                                        target_size: int = 200,
                                        ) -> List[float]:
        """Collect all pairwise distances between 2D distribution grids without averaging.
        
        Args:
            list_of_grids: List of 2D arrays containing the distributions.
            method: Method to use for distance calculation ('sliced' or 'sinkhorn').
            reg: Regularization parameter for Sinkhorn algorithm.
            num_iter: Number of iterations for Sinkhorn algorithm.
            num_projections: Number of projections for sliced Wasserstein distance.
            target_size: Size of the target grid for standardization.
            
        Returns:
            List of all pairwise distances between grids.
        """
        # Create a distance function that uses the selected method
        def distance_function(grid1, grid2):
            # Standardize grid sizes first
            grid1_std = self._transform_grid_to_standard_size(grid1, target_size)
            grid2_std = self._transform_grid_to_standard_size(grid2, target_size)
            
            # Calculate distance using selected method
            return self._2d_wasserstein_distance(
                grid1_std, grid2_std, 
                method=method,
                reg=reg,
                num_iter=num_iter,
                num_projections=num_projections,
                target_size=target_size
            )
        
        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(list_of_grids, distance_function=distance_function)
        
        # Return the raw distances
        return distance_matrix['raw_distances']
        
    def collect_pairwise_scatter_distribution_distances(self, 
                                        list_of_distributions: List[np.ndarray]) -> List[float]:
        """Collect all pairwise distances between scatter distributions without averaging.
        
        Args:
            list_of_distributions: List of arrays containing the scatter distributions.
            
        Returns:
            List of all pairwise distances between distributions.
        """
       
        
        # Calculate the pairwise distance matrix
        distance_matrix = self.pairwise_distance_matrix(list_of_distributions, distance_function=scipy.stats.wasserstein_distance_nd)
        
        # Return the raw distances
        return distance_matrix['raw_distances']