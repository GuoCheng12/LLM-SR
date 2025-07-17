#!/usr/bin/env python3
"""
Generate clean data with time points aligned to noisy data for fair comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def true_function(t, x, v):
    """True oscillator function: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)"""
    return 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)

def load_noisy_data(filepath):
    """Load noisy data from CSV file."""
    return pd.read_csv(filepath)

def generate_clean_data_with_time_alignment(time_points, seed=42):
    """
    Generate clean data using the exact same time points as noisy data.
    
    Args:
        time_points: time array from noisy data
        seed: random seed for reproducibility
    
    Returns:
        DataFrame with clean data
    """
    np.random.seed(seed)
    n_points = len(time_points)
    
    # Generate x, v with same distributions as original
    x = np.random.normal(0.5, 1.0, n_points)  # N(0.5, 1)
    v = np.random.normal(0.5, np.sqrt(0.5), n_points)  # N(0.5, 0.5)
    
    # Compute true acceleration (no noise)
    a = true_function(time_points, x, v)
    
    # Create DataFrame
    clean_df = pd.DataFrame({
        't': time_points,
        'x_clean': x,
        'v_clean': v,
        'a_clean': a
    })
    
    return clean_df

def create_comparison_visualization(clean_data, noisy_data, output_path):
    """
    Create side-by-side comparison of clean vs noisy data.
    
    Args:
        clean_data: DataFrame with clean data
        noisy_data: DataFrame with noisy data
        output_path: path to save the plot
    """
    # Set up the figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Clean vs Noisy Oscillator Data Comparison', fontsize=16, fontweight='bold')
    
    # Subsample for better visualization (every 20th point)
    step = 20
    clean_sub = clean_data.iloc[::step].copy()
    noisy_sub = noisy_data.iloc[::step].copy()
    
    # Left plot: Clean data
    ax1.plot(clean_sub['t'], clean_sub['x_clean'], 'b-', alpha=0.8, label='Position (x)', linewidth=2)
    ax1.plot(clean_sub['t'], clean_sub['v_clean'], 'g-', alpha=0.8, label='Velocity (v)', linewidth=2)
    ax1.plot(clean_sub['t'], clean_sub['a_clean'], 'r-', alpha=0.8, label='Acceleration (a)', linewidth=2)
    
    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Values', fontsize=12)
    ax1.set_title('Clean Data (No Noise)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-50, 10)  # Set consistent y-axis limits
    
    # Right plot: Noisy data
    ax2.plot(noisy_sub['t'], noisy_sub['datax'], 'b-', alpha=0.8, label='Position (x)', linewidth=2)
    ax2.plot(noisy_sub['t'], noisy_sub['datav'], 'g-', alpha=0.8, label='Velocity (v)', linewidth=2)
    ax2.plot(noisy_sub['t'], noisy_sub['dataa'], 'r-', alpha=0.8, label='Acceleration (a)', linewidth=2)
    
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Values', fontsize=12)
    ax2.set_title(f'Noisy Data (Group 1)\nσ_x={noisy_sub["sigma_x"].iloc[0]:.3f}, σ_v={noisy_sub["sigma_v"].iloc[0]:.3f}, σ_a={noisy_sub["sigma_a"].iloc[0]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-50, 10)  # Set consistent y-axis limits
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.show()

def create_detailed_comparison(clean_data, noisy_data, output_path):
    """
    Create a detailed 2x2 comparison showing individual variables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Clean vs Noisy Data Comparison', fontsize=16, fontweight='bold')
    
    # Subsample for better visualization
    step = 15
    clean_sub = clean_data.iloc[::step].copy()
    noisy_sub = noisy_data.iloc[::step].copy()
    
    # Plot 1: Position comparison
    axes[0, 0].plot(clean_sub['t'], clean_sub['x_clean'], 'b-', alpha=0.8, label='Clean', linewidth=2)
    axes[0, 0].plot(noisy_sub['t'], noisy_sub['datax'], 'r--', alpha=0.7, label='Noisy', linewidth=2)
    axes[0, 0].set_title('Position (x)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Velocity comparison
    axes[0, 1].plot(clean_sub['t'], clean_sub['v_clean'], 'g-', alpha=0.8, label='Clean', linewidth=2)
    axes[0, 1].plot(noisy_sub['t'], noisy_sub['datav'], 'r--', alpha=0.7, label='Noisy', linewidth=2)
    axes[0, 1].set_title('Velocity (v)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Acceleration comparison
    axes[1, 0].plot(clean_sub['t'], clean_sub['a_clean'], 'orange', alpha=0.8, label='Clean', linewidth=2)
    axes[1, 0].plot(noisy_sub['t'], noisy_sub['dataa'], 'r--', alpha=0.7, label='Noisy', linewidth=2)
    axes[1, 0].set_title('Acceleration (a)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (t)')
    axes[1, 0].set_ylabel('Acceleration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty over time
    axes[1, 1].plot(noisy_sub['t'], noisy_sub['sigma_total'], 'purple', alpha=0.8, linewidth=2, label='Total uncertainty')
    axes[1, 1].axhline(y=noisy_sub['sigma_a'].iloc[0], color='red', linestyle=':', alpha=0.7, label='Direct noise σ_a')
    axes[1, 1].set_title('Uncertainty over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (t)')
    axes[1, 1].set_ylabel('Uncertainty')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed plot
    detailed_output = output_path.with_name(output_path.stem + '_detailed.png')
    plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison plot saved to: {detailed_output}")
    
    plt.show()

def create_phase_space_comparison(clean_data, noisy_data, output_path):
    """
    Create phase space (x vs v) comparison between clean and noisy data.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Phase Space Comparison: Position vs Velocity', fontsize=16, fontweight='bold')
    
    # Subsample for better visualization
    step = 10
    clean_sub = clean_data.iloc[::step].copy()
    noisy_sub = noisy_data.iloc[::step].copy()
    
    # Left plot: Clean phase space
    scatter1 = ax1.scatter(clean_sub['x_clean'], clean_sub['a_clean'], 
                         c=clean_sub['t'], cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.set_ylabel('Velocity (v)', fontsize=12)
    ax1.set_title('Clean Data Phase Space', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add colorbar for time
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Time (t)', rotation=270, labelpad=20)
    
    # Right plot: Noisy phase space
    scatter2 = ax2.scatter(noisy_sub['datax'], noisy_sub['dataa'], 
                         c=noisy_sub['t'], cmap='viridis', alpha=0.6, s=20)
    ax2.set_xlabel('Position (x)', fontsize=12)
    ax2.set_ylabel('Velocity (v)', fontsize=12)
    ax2.set_title(f'Noisy Data Phase Space\n(σ_x={noisy_sub["sigma_x"].iloc[0]:.3f}, σ_v={noisy_sub["sigma_v"].iloc[0]:.3f})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Add colorbar for time
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Time (t)', rotation=270, labelpad=20)
    
    # Make axes limits consistent
    x_min = min(clean_sub['x_clean'].min(), noisy_sub['datax'].min())
    x_max = max(clean_sub['x_clean'].max(), noisy_sub['datax'].max())
    v_min = min(clean_sub['v_clean'].min(), noisy_sub['datav'].min())
    v_max = max(clean_sub['v_clean'].max(), noisy_sub['datav'].max())
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(v_min, v_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(v_min, v_max)
    
    plt.tight_layout()
    
    # Save the phase space plot
    phase_output = output_path.with_name(output_path.stem + '_phase_space.png')
    plt.savefig(phase_output, dpi=300, bbox_inches='tight')
    print(f"Phase space comparison plot saved to: {phase_output}")
    
    plt.show()

def print_comparison_stats(clean_data, noisy_data):
    """Print statistical comparison between clean and noisy data."""
    print("\n" + "="*60)
    print("ALIGNED DATA COMPARISON STATISTICS")
    print("="*60)
    
    print(f"Data points: {len(clean_data):,} (both datasets)")
    print(f"Time range: [{clean_data['t'].min():.3f}, {clean_data['t'].max():.3f}]")
    print(f"Time points match: {np.allclose(clean_data['t'].values, noisy_data['t'].values)}")
    
    # Noise characteristics
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
    
    # Correlation analysis
    print(f"\nCorrelation analysis (first 1000 points):")
    n_corr = min(1000, len(clean_data))
    corr_x = np.corrcoef(clean_data['x_clean'].iloc[:n_corr], noisy_data['datax'].iloc[:n_corr])[0,1]
    corr_v = np.corrcoef(clean_data['v_clean'].iloc[:n_corr], noisy_data['datav'].iloc[:n_corr])[0,1]
    corr_a = np.corrcoef(clean_data['a_clean'].iloc[:n_corr], noisy_data['dataa'].iloc[:n_corr])[0,1]
    
    print(f"  Position correlation: {corr_x:.3f}")
    print(f"  Velocity correlation: {corr_v:.3f}")
    print(f"  Acceleration correlation: {corr_a:.3f}")

def main():
    """Main function to generate aligned clean data and create visualizations."""
    print("Generating time-aligned clean data and visualizations...")
    
    # Paths
    data_dir = Path('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise')
    noisy_file = data_dir / 'train' / 'group_1.csv'
    clean_file = data_dir / 'clean_aligned.csv'
    plot_file = data_dir / 'clean_vs_noisy_comparison.png'
    
    # Load noisy data
    print("1. Loading noisy data...")
    noisy_data = load_noisy_data(noisy_file)
    
    # Generate time-aligned clean data
    print("2. Generating time-aligned clean data...")
    clean_data = generate_clean_data_with_time_alignment(noisy_data['t'].values, seed=42)
    
    # Save clean data
    print("3. Saving clean data...")
    clean_data.to_csv(clean_file, index=False)
    print(f"   Clean data saved to: {clean_file}")
    
    # Print comparison statistics
    print_comparison_stats(clean_data, noisy_data)
    
    # Create visualizations
    print("\n4. Creating side-by-side comparison...")
    create_comparison_visualization(clean_data, noisy_data, plot_file)
    
    print("5. Creating detailed comparison...")
    create_detailed_comparison(clean_data, noisy_data, plot_file)
    
    print("6. Creating phase space comparison...")
    create_phase_space_comparison(clean_data, noisy_data, plot_file)
    
    print("\nVisualization complete!")
    print(f"Files generated:")
    print(f"  - {clean_file}")
    print(f"  - {plot_file}")
    print(f"  - {plot_file.with_name(plot_file.stem + '_detailed.png')}")
    print(f"  - {plot_file.with_name(plot_file.stem + '_phase_space.png')}")

if __name__ == "__main__":
    main()