#!/usr/bin/env python3
"""
Simple noise analysis without matplotlib dependencies.
Generate clean sample and compare with noisy data.
"""

import numpy as np
import csv
from pathlib import Path

def true_function(t, x, v):
    """True oscillator function: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)"""
    return 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)

def generate_clean_sample(n_points=10000, t_range=(0, 50), seed=42):
    """Generate clean sample data."""
    np.random.seed(seed)
    
    # Sample inputs (same distributions as noisy data)
    t = np.random.uniform(t_range[0], t_range[1], n_points)
    x = np.random.normal(0.5, 1.0, n_points)  # N(0.5, 1)
    v = np.random.normal(0.5, np.sqrt(0.5), n_points)  # N(0.5, 0.5)
    
    # Compute true acceleration
    a = true_function(t, x, v)
    
    # Sort by time
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    x_sorted = x[sort_idx]
    v_sorted = v[sort_idx]
    a_sorted = a[sort_idx]
    
    return t_sorted, x_sorted, v_sorted, a_sorted

def load_noisy_data(filepath):
    """Load noisy data from CSV file."""
    data = {'t': [], 'datax': [], 'datav': [], 'dataa': [], 
            'sigma_x': [], 'sigma_v': [], 'sigma_a': [], 'sigma_total': []}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['t'].append(float(row['t']))
            data['datax'].append(float(row['datax']))
            data['datav'].append(float(row['datav']))
            data['dataa'].append(float(row['dataa']))
            data['sigma_x'].append(float(row['sigma_x']))
            data['sigma_v'].append(float(row['sigma_v']))
            data['sigma_a'].append(float(row['sigma_a']))
            data['sigma_total'].append(float(row['sigma_total']))
    
    return {k: np.array(v) for k, v in data.items()}

def save_clean_data(t, x, v, a, filepath):
    """Save clean data to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_clean', 'v_clean', 'a_clean'])
        for i in range(len(t)):
            writer.writerow([t[i], x[i], v[i], a[i]])

def analyze_noise(clean_data, noisy_data):
    """Analyze noise characteristics."""
    print("="*60)
    print("NOISE ANALYSIS: Clean vs Noisy Data")
    print("="*60)
    
    # Basic statistics
    print(f"Data points: {len(clean_data[0]):,} (clean) vs {len(noisy_data['t']):,} (noisy)")
    
    # Time range
    print(f"Time range: [{clean_data[0].min():.3f}, {clean_data[0].max():.3f}] (clean)")
    print(f"Time range: [{noisy_data['t'].min():.3f}, {noisy_data['t'].max():.3f}] (noisy)")
    
    # Noise scales
    print(f"\nNoise scales (Group 1):")
    print(f"  σ_x = {noisy_data['sigma_x'][0]:.4f}")
    print(f"  σ_v = {noisy_data['sigma_v'][0]:.4f}")
    print(f"  σ_a = {noisy_data['sigma_a'][0]:.4f}")
    
    # Variable statistics
    t_clean, x_clean, v_clean, a_clean = clean_data
    
    print(f"\nVariable ranges:")
    print(f"  Position (x): [{x_clean.min():.3f}, {x_clean.max():.3f}] (clean)")
    print(f"  Position (x): [{noisy_data['datax'].min():.3f}, {noisy_data['datax'].max():.3f}] (noisy)")
    print(f"  Velocity (v): [{v_clean.min():.3f}, {v_clean.max():.3f}] (clean)")
    print(f"  Velocity (v): [{noisy_data['datav'].min():.3f}, {noisy_data['datav'].max():.3f}] (noisy)")
    print(f"  Acceleration (a): [{a_clean.min():.3f}, {a_clean.max():.3f}] (clean)")
    print(f"  Acceleration (a): [{noisy_data['dataa'].min():.3f}, {noisy_data['dataa'].max():.3f}] (noisy)")
    
    # Uncertainty statistics
    print(f"\nUncertainty statistics:")
    print(f"  Total uncertainty: [{noisy_data['sigma_total'].min():.3f}, {noisy_data['sigma_total'].max():.3f}]")
    print(f"  Mean total uncertainty: {noisy_data['sigma_total'].mean():.3f}")
    print(f"  Std total uncertainty: {noisy_data['sigma_total'].std():.3f}")
    
    # Analyze noise (for first 1000 points)
    n_analyze = min(1000, len(t_clean), len(noisy_data['t']))
    
    noise_x = noisy_data['datax'][:n_analyze] - x_clean[:n_analyze]
    noise_v = noisy_data['datav'][:n_analyze] - v_clean[:n_analyze]
    noise_a = noisy_data['dataa'][:n_analyze] - a_clean[:n_analyze]
    
    print(f"\nNoise statistics (first {n_analyze} points):")
    print(f"  Position noise: mean={noise_x.mean():.4f}, std={noise_x.std():.4f}")
    print(f"  Velocity noise: mean={noise_v.mean():.4f}, std={noise_v.std():.4f}")
    print(f"  Acceleration noise: mean={noise_a.mean():.4f}, std={noise_a.std():.4f}")
    
    # Expected vs actual noise standard deviations
    print(f"\nExpected vs Actual noise std:")
    print(f"  Position: expected={noisy_data['sigma_x'][0]:.4f}, actual={noise_x.std():.4f}")
    print(f"  Velocity: expected={noisy_data['sigma_v'][0]:.4f}, actual={noise_v.std():.4f}")
    print(f"  Acceleration: expected={noisy_data['sigma_a'][0]:.4f}, actual={noise_a.std():.4f}")
    
def save_sample_comparison(clean_data, noisy_data, output_path, n_samples=100):
    """Save a sample comparison of clean vs noisy data points."""
    t_clean, x_clean, v_clean, a_clean = clean_data
    
    # Take every nth point for sampling
    step = len(t_clean) // n_samples
    indices = np.arange(0, len(t_clean), step)[:n_samples]
    
    filepath = Path(output_path) / 'sample_comparison.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_clean', 'v_clean', 'a_clean', 'x_noisy', 'v_noisy', 'a_noisy', 
                        'sigma_x', 'sigma_v', 'sigma_a', 'sigma_total'])
        
        for i in indices:
            if i < len(noisy_data['t']):
                writer.writerow([
                    t_clean[i], x_clean[i], v_clean[i], a_clean[i],
                    noisy_data['datax'][i], noisy_data['datav'][i], noisy_data['dataa'][i],
                    noisy_data['sigma_x'][i], noisy_data['sigma_v'][i], noisy_data['sigma_a'][i],
                    noisy_data['sigma_total'][i]
                ])
    
    print(f"\nSample comparison saved to: {filepath}")

def main():
    """Main function."""
    print("Generating clean vs noisy data analysis...")
    
    # Paths
    data_dir = Path('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise')
    noisy_file = data_dir / 'train' / 'group_1.csv'
    clean_file = data_dir / 'clean_sample.csv'
    
    # Generate clean sample data
    print("1. Generating clean sample data...")
    clean_data = generate_clean_sample(n_points=10000, t_range=(0, 50), seed=42)
    
    # Load noisy data
    print("2. Loading noisy data...")
    noisy_data = load_noisy_data(noisy_file)
    
    # Save clean data
    print("3. Saving clean data...")
    save_clean_data(*clean_data, clean_file)
    print(f"   Clean sample saved to: {clean_file}")
    
    # Analyze noise
    print("4. Analyzing noise characteristics...")
    analyze_noise(clean_data, noisy_data)
    
    # Save sample comparison
    print("5. Saving sample comparison...")
    save_sample_comparison(clean_data, noisy_data, data_dir, n_samples=100)
    
    print("\nAnalysis complete!")
    print(f"Files generated:")
    print(f"  - {clean_file}")
    print(f"  - {data_dir / 'sample_comparison.csv'}")

if __name__ == "__main__":
    main()