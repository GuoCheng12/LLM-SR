#!/usr/bin/env python3
"""
Generate oscillator2 noise dataset for LLM-SR robustness testing.

This script generates noise-robust datasets for testing LLM symbolic regression
capabilities on oscillator systems with additive Gaussian noise.

True function: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)

Dataset structure:
- train/: 8 groups, 10,000 points each (total 80,000 points)
- test_id/: 4 groups, 10,000 points each (total 40,000 points)  
- test_ood/: 3 groups, 10,000 points each (total 30,000 points)

Each group has independent noise scales sigma_x, sigma_v, sigma_a ~ U(0.01, 0.1)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def true_function(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    True oscillator function: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)
    
    Args:
        t: time array
        x: position array
        v: velocity array
    
    Returns:
        acceleration array
    """
    return 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)

def compute_partial_derivatives(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute partial derivatives of the true function for uncertainty propagation.
    
    df/dx = -v - 5*exp(0.5*x) - 2.5*x*exp(0.5*x)
    df/dv = -1.5*v^2 - x
    
    Args:
        t: time array
        x: position array  
        v: velocity array
        
    Returns:
        Tuple of (df_dx, df_dv)
    """
    df_dx = -v - 5 * np.exp(0.5 * x) - 2.5 * x * np.exp(0.5 * x)
    df_dv = -1.5 * v**2 - x
    return df_dx, df_dv

def generate_clean_data(n_points: int, t_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
    """
    Generate clean (noise-free) data samples.
    
    Args:
        n_points: number of data points
        t_range: time range (t_min, t_max)
        
    Returns:
        Dictionary with keys 't', 'x', 'v', 'a'
    """
    # Sample inputs
    t = np.random.uniform(t_range[0], t_range[1], n_points)
    x = np.random.normal(0.5, 1.0, n_points)  # N(0.5, 1)
    v = np.random.normal(0.5, np.sqrt(0.5), n_points)  # N(0.5, 0.5)
    
    # Compute true acceleration
    a = true_function(t, x, v)
    
    return {'t': t, 'x': x, 'v': v, 'a': a}

def add_group_noise(data: Dict[str, np.ndarray], sigma_x: float, sigma_v: float, sigma_a: float) -> Dict[str, np.ndarray]:
    """
    Add group-level additive Gaussian noise to data.
    
    Args:
        data: clean data dictionary
        sigma_x: noise scale for position
        sigma_v: noise scale for velocity
        sigma_a: noise scale for acceleration
        
    Returns:
        Dictionary with noisy observations and uncertainties
    """
    n_points = len(data['t'])
    
    # Generate noise samples
    eps_x = np.random.normal(0, 1, n_points)
    eps_v = np.random.normal(0, 1, n_points)
    eps_a = np.random.normal(0, 1, n_points)
    
    # Add noise to observations (time remains noise-free)
    noisy_data = {
        't': data['t'],  # time is noise-free
        'datax': data['x'] + sigma_x * eps_x,
        'datav': data['v'] + sigma_v * eps_v,
        'dataa': data['a'] + sigma_a * eps_a,
        'sigma_x': np.full(n_points, sigma_x),
        'sigma_v': np.full(n_points, sigma_v),
        'sigma_a': np.full(n_points, sigma_a),
    }
    
    # Compute total uncertainty via error propagation
    df_dx, df_dv = compute_partial_derivatives(data['t'], data['x'], data['v'])
    sigma_total = np.sqrt(
        sigma_a**2 + 
        (np.abs(df_dx) * sigma_x)**2 + 
        (np.abs(df_dv) * sigma_v)**2
    )
    noisy_data['sigma_total'] = sigma_total
    
    return noisy_data

def generate_dataset_split(split_name: str, n_groups: int, points_per_group: int, 
                          t_range: Tuple[float, float], noise_range: Tuple[float, float],
                          output_dir: Path) -> None:
    """
    Generate a complete dataset split (train/test_id/test_ood).
    
    Args:
        split_name: name of the split ('train', 'test_id', 'test_ood')
        n_groups: number of groups in this split
        points_per_group: number of points per group
        t_range: time range for sampling
        noise_range: noise scale range (sigma_min, sigma_max)
        output_dir: output directory path
    """
    logger.info(f"Generating {split_name} dataset: {n_groups} groups x {points_per_group} points")
    
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for group_idx in range(1, n_groups + 1):
        logger.info(f"  Processing group {group_idx}/{n_groups}")
        
        # Generate clean data
        clean_data = generate_clean_data(points_per_group, t_range)
        
        # Sample group-level noise scales
        sigma_x = np.random.uniform(noise_range[0], noise_range[1])
        sigma_v = np.random.uniform(noise_range[0], noise_range[1])
        sigma_a = np.random.uniform(noise_range[0], noise_range[1])
        
        logger.info(f"    Noise scales: σ_x={sigma_x:.4f}, σ_v={sigma_v:.4f}, σ_a={sigma_a:.4f}")
        
        # Add noise
        noisy_data = add_group_noise(clean_data, sigma_x, sigma_v, sigma_a)
        
        # Save to CSV with time-based sorting
        df = pd.DataFrame(noisy_data)
        df_sorted = df.sort_values('t').reset_index(drop=True)
        output_file = split_dir / f"group_{group_idx}.csv"
        df_sorted.to_csv(output_file, index=False)
        
        logger.info(f"    Saved to {output_file}")

def main():
    """Main function to generate all dataset splits."""
    # Configuration
    output_dir = Path("/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise")
    points_per_group = 10000
    
    # Dataset specifications
    splits_config = [
        {
            'name': 'train',
            'n_groups': 8,
            't_range': (30, 50),
            'noise_range': (0.01, 0.1)
        },
        {
            'name': 'test_id', 
            'n_groups': 4,
            't_range': (30, 50),
            'noise_range': (0.01, 0.1)
        },
        {
            'name': 'test_ood',
            'n_groups': 3,
            't_range': (0, 20),  # Different time range for OOD
            'noise_range': (0.05, 0.15)  # Higher noise for OOD
        }
    ]
    
    # Generate all splits
    logger.info("Starting noise dataset generation...")
    for config in splits_config:
        generate_dataset_split(
            split_name=config['name'],
            n_groups=config['n_groups'],
            points_per_group=points_per_group,
            t_range=config['t_range'],
            noise_range=config['noise_range'],
            output_dir=output_dir
        )
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    total_points = sum(config['n_groups'] * points_per_group for config in splits_config)
    logger.info(f"Dataset generation complete!")
    logger.info(f"Total points generated: {total_points:,}")
    logger.info(f"Dataset structure:")
    for config in splits_config:
        points = config['n_groups'] * points_per_group
        logger.info(f"  {config['name']}: {config['n_groups']} groups x {points_per_group:,} points = {points:,} total")
    
    # Save metadata
    metadata = {
        'true_function': 'a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)',
        'noise_model': 'Additive Gaussian noise on x, v, a (time t is noise-free)',
        'input_distributions': {
            'x': 'N(0.5, 1.0)',
            'v': 'N(0.5, 0.5)',
            't_train_test_id': 'U(30, 50)',
            't_test_ood': 'U(0, 20)'
        },
        'noise_ranges': {
            'train_test_id': 'U(0.01, 0.1)',
            'test_ood': 'U(0.05, 0.15)'
        },
        'total_points': total_points,
        'splits': splits_config
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()