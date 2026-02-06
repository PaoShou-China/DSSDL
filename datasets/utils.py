"""
datasets/utils.py - Utility functions for dataset handling and processing.

This module provides functions for:
- Clearing existing dataset files
- Loading and processing dataset files
- Calculating normalization parameters
- Standardizing and inverse normalizing data
- Generic dataset loading for different systems
"""

import numpy as np
import os

def clear_dataset_files(system_name):
    """
    Clear existing dataset files for a system.
    
    Args:
        system_name: Name of the system to clear dataset files for
    """
    base_dir = f"datasets/{system_name}"
    dataset_types = ["train", "support", "query"]
    data_types = ["features", "labels"]
    print(f"清除 {system_name} 系统的现有数据集文件...")
    deleted_count = 0
    for dataset_type in dataset_types:
        for data_type in data_types:
            dir_path = os.path.join(base_dir, dataset_type, data_type)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(dir_path, file)
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                        deleted_count += 1
    if deleted_count == 0:
        print(f"没有找到需要删除的 {system_name} 数据集文件")
    else:
        print(f"共删除了 {deleted_count} 个 {system_name} 数据集文件")

def get_file_list(directory):
    """
    Get sorted list of .txt files in a directory.
    
    Args:
        directory: Directory to search for files
        
    Returns:
        List of file paths
        
    Raises:
        FileNotFoundError: If directory does not exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(".txt")
    ])

def load_file_list(file_paths):
    """
    Load list of files as numpy arrays.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of numpy arrays
    """
    return [np.loadtxt(path, ndmin=2) for path in file_paths]

def calculate_normalization_params(data_list):
    """
    Calculate normalization parameters (mean and std) for data.
    
    Args:
        data_list: List of numpy arrays to calculate parameters from
        
    Returns:
        Tuple of (mean, std)
    """
    all_data = np.concatenate(data_list, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std[std == 0] = 1e-9
    return mean, std

def standardize(data, mean, std):
    """
    Standardize data using mean and std.
    
    Args:
        data: Data to standardize
        mean: Mean for standardization
        std: Standard deviation for standardization
        
    Returns:
        Standardized data
    """
    return (data - mean) / std

def inverse_normalize_data(data, mean, std):
    """
    Inverse normalize data using mean and std.
    
    Args:
        data: Data to inverse normalize
        mean: Mean used for standardization
        std: Standard deviation used for standardization
        
    Returns:
        Inverse normalized data
    """
    return data * std + mean

def get_datasets_generic(system_path, use_normalization=True):
    """
    Generic function to load datasets for any system.
    
    Args:
        system_path: Path to the system directory
        use_normalization: Whether to use data normalization
        
    Returns:
        Dictionary containing the dataset and metadata
    """
    system_dt = 0.01
    try:
        system_name = os.path.basename(system_path)
        module = __import__(f'datasets.{system_name}', fromlist=['dt'])
        system_dt = getattr(module, 'dt', system_dt)
    except (ImportError, AttributeError):
        pass
    
    # Get file paths for all splits and data types
    data_paths = {
        "train": {
            "features": get_file_list(os.path.join(system_path, "train", "features")),
            "labels": get_file_list(os.path.join(system_path, "train", "labels")),
        },
        "support": {
            "features": get_file_list(os.path.join(system_path, "support", "features")),
            "labels": get_file_list(os.path.join(system_path, "support", "labels")),
        },
        "query": {
            "features": get_file_list(os.path.join(system_path, "query", "features")),
            "labels": get_file_list(os.path.join(system_path, "query", "labels")),
        }
    }
    
    # Process datasets
    raw_datasets = {}
    normalized_datasets = {}
    sample_counts = {}
    file_counts = {}
    calculated_normalization = {}
    input_mean, input_std = None, None
    output_mean, output_std = None, None
    
    for split in ['train', 'support', 'query']:
        feat_paths = data_paths[split]['features']
        label_paths = data_paths[split]['labels']
        X_list = load_file_list(feat_paths)
        y_list = load_file_list(label_paths)
        
        # Calculate normalization parameters from training data
        if split == 'train':
            input_mean, input_std = calculate_normalization_params(X_list)
            output_mean, output_std = calculate_normalization_params(y_list)
            calculated_normalization = {
                "input": {"mean": input_mean.tolist(), "std": input_std.tolist()},
                "output": {"mean": output_mean.tolist(), "std": output_std.tolist()}
            }
        
        if input_mean is None or output_mean is None:
            raise RuntimeError("Normalization parameters not calculated. 'train' split must be processed first.")
        
        # Store raw datasets
        raw_datasets[split] = (X_list, y_list)
        
        # Store normalized datasets if requested
        if use_normalization:
            X_norm_list = [standardize(X, input_mean, input_std) for X in X_list]
            y_norm_list = [standardize(y, output_mean, output_std) for y in y_list]
            normalized_datasets[split] = (X_norm_list, y_norm_list)
        
        # Calculate statistics
        total_samples = sum(X.shape[0] for X in X_list)
        assert len(feat_paths) == len(label_paths)
        file_counts[split] = len(feat_paths)
        sample_counts[split] = total_samples
    
    # Get input and output dimensions
    input_dim = X_list[0].shape[1] if X_list else 0
    output_dim = y_list[0].shape[1] if y_list else 0
    
    # Create info dictionary
    info = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'dt': system_dt,
        'normalization': calculated_normalization,
        'file_counts': file_counts,
        'sample_counts': sample_counts
    }
    
    # Create result dictionary
    result = {
        'raw': raw_datasets,
        'info': info
    }
    
    if use_normalization:
        result['normalized'] = normalized_datasets
    
    return result