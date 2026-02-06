"""
datasets/__init__.py - Main datasets module for system identification.

This module provides functionality to:
- Load and switch between different system datasets
- Get dataset dictionaries for training and evaluation
- List available systems
- Handle system-specific configurations
"""

import os
import importlib
from .utils import get_datasets_generic

CURRENT_SYSTEM = 'pendulum_real'

def get_system_module(system_name):
    """
    Get the module for a specific system.
    
    Args:
        system_name: Name of the system
        
    Returns:
        Module for the specified system
        
    Raises:
        ImportError: If the system module cannot be imported
    """
    try:
        module = importlib.import_module(f'.{system_name}', package='datasets')
        return module
    except ImportError as e:
        raise ImportError(f"无法导入系统 '{system_name}': {e}")

def set_current_system(system_name):
    """
    Set the current system to use.
    
    Args:
        system_name: Name of the system to set as current
    """
    global CURRENT_SYSTEM, dataset_dict, f, dt, slow_params_eval
    CURRENT_SYSTEM = system_name
    system_module = get_system_module(system_name)
    dataset_dict = system_module.dataset_dict
    f = system_module.f
    dt = system_module.dt
    slow_params_eval = system_module.slow_params_eval
    print(f"已切换到系统: {system_name}")

def get_raw_dataset_dict():
    """
    Get the raw dataset dictionary for the current system.
    
    Returns:
        Raw dataset dictionary
    """
    system_module = get_system_module(CURRENT_SYSTEM)
    return system_module.dataset_dict

def get_datasets(use_normalization=True):
    """
    Get datasets for the current system.
    
    Args:
        use_normalization: Whether to use data normalization
        
    Returns:
        Dataset dictionary
    """
    system_module = get_system_module(CURRENT_SYSTEM)
    return system_module.get_datasets(use_normalization)

def list_available_systems():
    """
    List all available systems in the datasets directory.
    
    Returns:
        List of available system names
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    systems = []
    for item in os.listdir(current_dir):
        system_path = os.path.join(current_dir, item)
        if (os.path.isdir(system_path) and 
            not item.startswith('__') and 
            not item.startswith('.') and
            os.path.exists(os.path.join(system_path, '__init__.py'))):
            systems.append(item)
    return systems

# Initialize default system
try:
    system_module = get_system_module(CURRENT_SYSTEM)
    dataset_dict = system_module.dataset_dict
    f = system_module.f
    dt = system_module.dt
    slow_params_eval = system_module.slow_params_eval
except ImportError as e:
    print(f"警告: 无法加载默认系统 '{CURRENT_SYSTEM}': {e}")
    print(f"可用系统: {list_available_systems()}")
    dataset_dict = None
    f = None
    dt = 0.01
    slow_params_eval = []