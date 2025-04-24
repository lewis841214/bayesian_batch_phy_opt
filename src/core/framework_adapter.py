from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from src.core.parameter_space import ParameterSpace

class FrameworkAdapter(ABC):
    """Base adapter class for converting between parameter spaces"""
    
    def __init__(self, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space
    
    @abstractmethod
    def to_framework_format(self):
        """Convert generic parameter space to framework-specific format"""
        pass
    
    @abstractmethod
    def from_framework_format(self, framework_params):
        """Convert framework-specific parameters to standard dictionary"""
        pass
    
    @abstractmethod
    def convert_results(self, framework_results):
        """Convert framework-specific results to standard format"""
        pass 