#!/usr/bin/env python3
"""
Visualize comparison between clean and noisy oscillator data.

This script generates a clean sample dataset and compares it with the noisy
train/group_1.csv to validate the noise addition process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def true_function(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    True oscillator function: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)
    """
    return 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)

def generate_clean_sample(n_points: int = 10000, t_range: tuple = (0, 50)) -> pd.DataFrame:
    """
    Generate clean (noise-free) sample data for comparison.
    
    Args:
        n_points: number of data points
        t_range: time range (t_min, t_max)
        
    Returns:
        DataFrame with clean data
    """
    # Sample inputs (same distributions as noisy data)
    np.random.seed(42)  # For reproducibility
    t = np.random.uniform(t_range[0], t_range[1], n_points)
    x = np.random.normal(0.5, 1.0, n_points)  # N(0.5, 1)
    v = np.random.normal(0.5, np.sqrt(0.5), n_points)  # N(0.5, 0.5)
    
    # Compute true acceleration
    a = true_function(t, x, v)
    
    # Create DataFrame and sort by time
    df = pd.DataFrame({
        't': t,
        'x_clean': x,
        'v_clean': v,
        'a_clean': a
    })
    
    return df.sort_values('t').reset_index(drop=True)

def load_noisy_data(filepath: str) -> pd.DataFrame:
    """Load noisy data from CSV file."""
    return pd.read_csv(filepath)

def create_comparison_plots(clean_data: pd.DataFrame, noisy_data: pd.DataFrame, 
                          output_path: str = '/Users/wuguocheng/workshop/LLM-SR/data/'):
    """
    Create comparison plots between clean and noisy data.
    
    Args:
        clean_data: DataFrame with clean data
        noisy_data: DataFrame with noisy data  
        output_path: path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clean vs Noisy Oscillator Data Comparison', fontsize=16, fontweight='bold')
    
    # Subsample for better visualization (every 10th point)
    step = 10
    clean_sub = clean_data.iloc[::step].copy()
    noisy_sub = noisy_data.iloc[::step].copy()
    
    # Plot 1: Position vs Time
    ax1 = axes[0, 0]
    ax1.plot(clean_sub['t'], clean_sub['x_clean'], 'b-', alpha=0.7, label='Clean x', linewidth=2)
    ax1.plot(noisy_sub['t'], noisy_sub['datax'], 'r.', alpha=0.5, label='Noisy x', markersize=3)
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position (x)')
    ax1.set_title('Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity vs Time
    ax2 = axes[0, 1]
    ax2.plot(clean_sub['t'], clean_sub['v_clean'], 'b-', alpha=0.7, label='Clean v', linewidth=2)
    ax2.plot(noisy_sub['t'], noisy_sub['datav'], 'r.', alpha=0.5, label='Noisy v', markersize=3)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Velocity (v)')
    ax2.set_title('Velocity vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Acceleration vs Time
    ax3 = axes[1, 0]
    ax3.plot(clean_sub['t'], clean_sub['a_clean'], 'b-', alpha=0.7, label='Clean a', linewidth=2)
    ax3.plot(noisy_sub['t'], noisy_sub['dataa'], 'r.', alpha=0.5, label='Noisy a', markersize=3)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Acceleration (a)')
    ax3.set_title('Acceleration vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Phase Space (x vs v)
    ax4 = axes[1, 1]
    ax4.plot(clean_sub['x_clean'], clean_sub['v_clean'], 'b-', alpha=0.7, label='Clean trajectory', linewidth=2)
    ax4.plot(noisy_sub['datax'], noisy_sub['datav'], 'r.', alpha=0.5, label='Noisy trajectory', markersize=3)
    ax4.set_xlabel('Position (x)')
    ax4.set_ylabel('Velocity (v)')
    ax4.set_title('Phase Space (x vs v)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'clean_vs_noisy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    
    plt.show()

def create_noise_analysis_plots(clean_data: pd.DataFrame, noisy_data: pd.DataFrame,
                               output_path: str = '/Users/wuguocheng/workshop/LLM-SR/data/'):
    """
    Create noise analysis plots showing noise characteristics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Noise Analysis: Clean vs Noisy Data', fontsize=16, fontweight='bold')
    
    # Compute noise (difference between noisy and clean)
    # Note: We need to interpolate clean data to match noisy data points
    # For simplicity, we'll use the first 1000 points for analysis
    n_analyze = 1000
    clean_analyze = clean_data.iloc[:n_analyze].copy()
    noisy_analyze = noisy_data.iloc[:n_analyze].copy()
    
    # Compute noise for each variable
    noise_x = noisy_analyze['datax'].values - clean_analyze['x_clean'].values
    noise_v = noisy_analyze['datav'].values - clean_analyze['v_clean'].values  
    noise_a = noisy_analyze['dataa'].values - clean_analyze['a_clean'].values
    
    # Plot 1: Noise distribution for x
    ax1 = axes[0, 0]
    ax1.hist(noise_x, bins=50, alpha=0.7, density=True, label=f'σ_x = {noisy_analyze["sigma_x"].iloc[0]:.4f}')
    ax1.set_xlabel('Noise in x')
    ax1.set_ylabel('Density')
    ax1.set_title('Position Noise Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise distribution for v
    ax2 = axes[0, 1]
    ax2.hist(noise_v, bins=50, alpha=0.7, density=True, label=f'σ_v = {noisy_analyze["sigma_v"].iloc[0]:.4f}')
    ax2.set_xlabel('Noise in v')
    ax2.set_ylabel('Density')
    ax2.set_title('Velocity Noise Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Noise distribution for a
    ax3 = axes[1, 0]
    ax3.hist(noise_a, bins=50, alpha=0.7, density=True, label=f'σ_a = {noisy_analyze["sigma_a"].iloc[0]:.4f}')
    ax3.set_xlabel('Noise in a')
    ax3.set_ylabel('Density')
    ax3.set_title('Acceleration Noise Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty comparison
    ax4 = axes[1, 1]
    ax4.plot(noisy_analyze['t'], noisy_analyze['sigma_total'], 'g-', alpha=0.7, label='Total uncertainty')
    ax4.axhline(y=noisy_analyze['sigma_a'].iloc[0], color='r', linestyle='--', alpha=0.7, label='Direct noise σ_a')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Uncertainty vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'noise_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Noise analysis plot saved to: {output_file}")
    
    plt.show()

def print_statistics(clean_data: pd.DataFrame, noisy_data: pd.DataFrame):
    """Print statistical comparison between clean and noisy data."""
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: CLEAN vs NOISY DATA")
    print("="*60)
    
    # Basic statistics
    print(f"\nData points: {len(clean_data):,} (clean) vs {len(noisy_data):,} (noisy)")
    
    # Time range
    print(f"Time range: [{clean_data['t'].min():.3f}, {clean_data['t'].max():.3f}] (clean)")
    print(f"Time range: [{noisy_data['t'].min():.3f}, {noisy_data['t'].max():.3f}] (noisy)")
    
    # Noise scales
    print(f"\nNoise scales (Group 1):")
    print(f"  σ_x = {noisy_data['sigma_x'].iloc[0]:.4f}")
    print(f"  σ_v = {noisy_data['sigma_v'].iloc[0]:.4f}")
    print(f"  σ_a = {noisy_data['sigma_a'].iloc[0]:.4f}")
    
    # Variable ranges
    print(f"\nVariable ranges:")
    print(f"  Position (x): [{clean_data['x_clean'].min():.3f}, {clean_data['x_clean'].max():.3f}] (clean)")
    print(f"  Position (x): [{noisy_data['datax'].min():.3f}, {noisy_data['datax'].max():.3f}] (noisy)")
    print(f"  Velocity (v): [{clean_data['v_clean'].min():.3f}, {clean_data['v_clean'].max():.3f}] (clean)")
    print(f"  Velocity (v): [{noisy_data['datav'].min():.3f}, {noisy_data['datav'].max():.3f}] (noisy)")
    print(f"  Acceleration (a): [{clean_data['a_clean'].min():.3f}, {clean_data['a_clean'].max():.3f}] (clean)")
    print(f"  Acceleration (a): [{noisy_data['dataa'].min():.3f}, {noisy_data['dataa'].max():.3f}] (noisy)")
    
    # Uncertainty statistics
    print(f"\nUncertainty statistics:")
    print(f"  Total uncertainty: [{noisy_data['sigma_total'].min():.3f}, {noisy_data['sigma_total'].max():.3f}]")
    print(f"  Mean total uncertainty: {noisy_data['sigma_total'].mean():.3f}")

def main():
    """Main function to generate comparison visualizations."""
    print("Generating clean vs noisy data comparison...")
    
    # Paths
    data_dir = Path('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise')
    noisy_file = data_dir / 'train' / 'group_1.csv'
    
    # Generate clean sample data
    print("1. Generating clean sample data...")
    clean_data = generate_clean_sample(n_points=10000, t_range=(0, 50))
    
    # Load noisy data
    print("2. Loading noisy data...")
    noisy_data = load_noisy_data(noisy_file)
    
    # Print statistics
    print_statistics(clean_data, noisy_data)
    
    # Create comparison plots
    print("\n3. Creating comparison plots...")
    create_comparison_plots(clean_data, noisy_data)
    
    # Create noise analysis plots
    print("4. Creating noise analysis plots...")
    create_noise_analysis_plots(clean_data, noisy_data)
    
    # Save clean data for reference
    clean_file = data_dir / 'clean_sample.csv'
    clean_data.to_csv(clean_file, index=False)
    print(f"\nClean sample data saved to: {clean_file}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()