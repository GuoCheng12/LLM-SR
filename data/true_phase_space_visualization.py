#!/usr/bin/env python3
"""
Generate true phase space visualization by numerically integrating the differential equation system.

The given equation: a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)
Can be written as a system of first-order ODEs:
  dx/dt = v
  dv/dt = a = 0.3*sin(t) - 0.5*v^3 - x*v - 5*x*exp(0.5*x)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('default')

def oscillator_system(t, y):
    """
    Define the oscillator system of ODEs.
    
    Args:
        t: time
        y: state vector [x, v]
    
    Returns:
        dydt: derivative vector [dx/dt, dv/dt]
    """
    x, v = y
    
    # Compute acceleration
    a = 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)
    
    # Return derivatives
    return [v, a]

def solve_clean_trajectory(x0, v0, t_span, t_eval):
    """
    Solve the clean (noise-free) trajectory using numerical integration.
    
    Args:
        x0: initial position
        v0: initial velocity
        t_span: time span [t_start, t_end]
        t_eval: time points to evaluate
    
    Returns:
        solution object with t, x, v arrays
    """
    # Initial conditions
    y0 = [x0, v0]
    
    # Solve the ODE system
    sol = solve_ivp(oscillator_system, t_span, y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-8, atol=1e-10)
    
    return sol

def add_noise_to_trajectory(t, x, v, noise_level=0.05):
    """
    Add noise to a clean trajectory.
    
    Args:
        t: time array
        x: position array
        v: velocity array
        noise_level: noise standard deviation
    
    Returns:
        noisy_x, noisy_v: noisy position and velocity arrays
    """
    np.random.seed(42)  # For reproducibility
    
    # Add Gaussian noise
    noise_x = np.random.normal(0, noise_level, len(x))
    noise_v = np.random.normal(0, noise_level, len(v))
    
    noisy_x = x + noise_x
    noisy_v = v + noise_v
    
    return noisy_x, noisy_v

def create_multiple_trajectories(n_trajectories=5, t_span=(0, 50), n_points=5000):
    """
    Create multiple trajectories with different initial conditions.
    
    Args:
        n_trajectories: number of trajectories
        t_span: time span
        n_points: number of time points
    
    Returns:
        list of trajectory solutions
    """
    trajectories = []
    
    # Different initial conditions
    initial_conditions = [
        (0.5, 0.5),    # Original
        (1.0, 0.0),    # Different x0
        (0.0, 1.0),    # Different v0
        (-0.5, 0.5),   # Negative x0
        (0.5, -0.5),   # Negative v0
    ]
    
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    for i, (x0, v0) in enumerate(initial_conditions[:n_trajectories]):
        print(f"  Generating trajectory {i+1}/{n_trajectories} with IC: x0={x0}, v0={v0}")
        sol = solve_clean_trajectory(x0, v0, t_span, t_eval)
        
        if sol.success:
            trajectories.append({
                't': sol.t,
                'x': sol.y[0],
                'v': sol.y[1],
                'x0': x0,
                'v0': v0
            })
        else:
            print(f"    Warning: Trajectory {i+1} failed to solve")
    
    return trajectories

def create_phase_space_comparison(trajectories, output_path, noise_level=0.05):
    """
    Create phase space comparison between clean and noisy trajectories.
    
    Args:
        trajectories: list of trajectory dictionaries
        output_path: path to save the plot
        noise_level: noise level for noisy trajectories
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Phase Space Comparison: Clean vs Noisy Trajectories', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        
        # Plot clean trajectory
        ax1.plot(traj['x'], traj['v'], color=color, alpha=0.8, linewidth=2,
                label=f"IC: x₀={traj['x0']}, v₀={traj['v0']}")
        
        # Add noise and plot noisy trajectory
        noisy_x, noisy_v = add_noise_to_trajectory(traj['t'], traj['x'], traj['v'], noise_level)
        ax2.plot(noisy_x, noisy_v, color=color, alpha=0.6, linewidth=1.5,
                label=f"IC: x₀={traj['x0']}, v₀={traj['v0']}")
        
        # Mark starting points
        ax1.scatter(traj['x'][0], traj['v'][0], color=color, s=100, marker='o', 
                   edgecolors='black', linewidth=2, zorder=5)
        ax2.scatter(noisy_x[0], noisy_v[0], color=color, s=100, marker='o', 
                   edgecolors='black', linewidth=2, zorder=5)
    
    # Configure left plot (clean)
    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.set_ylabel('Velocity (v)', fontsize=12)
    ax1.set_title('Clean Phase Space', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Configure right plot (noisy)
    ax2.set_xlabel('Position (x)', fontsize=12)
    ax2.set_ylabel('Velocity (v)', fontsize=12)
    ax2.set_title(f'Noisy Phase Space (σ = {noise_level:.3f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Set consistent axis limits
    all_x = np.concatenate([traj['x'] for traj in trajectories])
    all_v = np.concatenate([traj['v'] for traj in trajectories])
    
    x_margin = 0.1 * (all_x.max() - all_x.min())
    v_margin = 0.1 * (all_v.max() - all_v.min())
    
    ax1.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax1.set_ylim(all_v.min() - v_margin, all_v.max() + v_margin)
    ax2.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax2.set_ylim(all_v.min() - v_margin, all_v.max() + v_margin)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Phase space comparison saved to: {output_path}")
    
    plt.show()

def create_single_trajectory_comparison(output_path, noise_level=0.05):
    """
    Create a detailed comparison of a single trajectory.
    """
    # Generate single trajectory
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 5000)
    
    print("Generating single trajectory for detailed comparison...")
    sol = solve_clean_trajectory(0.5, 0.5, t_span, t_eval)
    
    if not sol.success:
        print("Failed to solve trajectory")
        return
    
    # Add noise
    noisy_x, noisy_v = add_noise_to_trajectory(sol.t, sol.y[0], sol.y[1], noise_level)
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Single Trajectory Analysis: Clean vs Noisy', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series - Position
    axes[0, 0].plot(sol.t, sol.y[0], 'b-', label='Clean', linewidth=2)
    axes[0, 0].plot(sol.t, noisy_x, 'r--', label='Noisy', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Position (x)')
    axes[0, 0].set_title('Position vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time series - Velocity
    axes[0, 1].plot(sol.t, sol.y[1], 'g-', label='Clean', linewidth=2)
    axes[0, 1].plot(sol.t, noisy_v, 'r--', label='Noisy', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Velocity (v)')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Phase space - Clean
    axes[1, 0].plot(sol.y[0], sol.y[1], 'b-', linewidth=2, alpha=0.8)
    axes[1, 0].scatter(sol.y[0][0], sol.y[1][0], color='green', s=100, marker='o', 
                      edgecolors='black', linewidth=2, zorder=5, label='Start')
    axes[1, 0].scatter(sol.y[0][-1], sol.y[1][-1], color='red', s=100, marker='s', 
                      edgecolors='black', linewidth=2, zorder=5, label='End')
    axes[1, 0].set_xlabel('Position (x)')
    axes[1, 0].set_ylabel('Velocity (v)')
    axes[1, 0].set_title('Clean Phase Space')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Phase space - Noisy
    axes[1, 1].plot(noisy_x, noisy_v, 'r-', linewidth=1.5, alpha=0.7)
    axes[1, 1].scatter(noisy_x[0], noisy_v[0], color='green', s=100, marker='o', 
                      edgecolors='black', linewidth=2, zorder=5, label='Start')
    axes[1, 1].scatter(noisy_x[-1], noisy_v[-1], color='red', s=100, marker='s', 
                      edgecolors='black', linewidth=2, zorder=5, label='End')
    axes[1, 1].set_xlabel('Position (x)')
    axes[1, 1].set_ylabel('Velocity (v)')
    axes[1, 1].set_title(f'Noisy Phase Space (σ = {noise_level:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed plot
    detailed_output = output_path.with_name(output_path.stem + '_detailed.png')
    plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison saved to: {detailed_output}")
    
    plt.show()

def save_trajectory_data(trajectories, output_dir):
    """
    Save trajectory data to CSV files.
    """
    for i, traj in enumerate(trajectories):
        # Create DataFrame
        df = pd.DataFrame({
            't': traj['t'],
            'x_clean': traj['x'],
            'v_clean': traj['v']
        })
        
        # Add noisy versions
        noisy_x, noisy_v = add_noise_to_trajectory(traj['t'], traj['x'], traj['v'])
        df['x_noisy'] = noisy_x
        df['v_noisy'] = noisy_v
        
        # Save to CSV
        filename = f"trajectory_{i+1}_x0_{traj['x0']}_v0_{traj['v0']}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Trajectory {i+1} saved to: {filepath}")

def main():
    """Main function to generate phase space visualizations."""
    print("Generating true phase space visualizations...")
    
    # Paths
    data_dir = Path('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise')
    plot_file = data_dir / 'true_phase_space_comparison.png'
    
    # Generate multiple trajectories
    print("1. Generating multiple trajectories...")
    trajectories = create_multiple_trajectories(n_trajectories=3, t_span=(0, 30), n_points=3000)
    
    if not trajectories:
        print("Error: No trajectories were successfully generated")
        return
    
    print(f"   Successfully generated {len(trajectories)} trajectories")
    
    # Create phase space comparison
    print("2. Creating phase space comparison...")
    create_phase_space_comparison(trajectories, plot_file, noise_level=0.05)
    
    # Create single trajectory detailed comparison
    print("3. Creating detailed single trajectory comparison...")
    create_single_trajectory_comparison(plot_file, noise_level=0.05)
    
    # Save trajectory data
    print("4. Saving trajectory data...")
    save_trajectory_data(trajectories, data_dir)
    
    print("\nPhase space visualization complete!")
    print(f"Files generated:")
    print(f"  - {plot_file}")
    print(f"  - {plot_file.with_name(plot_file.stem + '_detailed.png')}")
    print(f"  - Multiple trajectory CSV files in {data_dir}")
    
    # Print trajectory statistics
    print(f"\nTrajectory statistics:")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i+1}: IC=(x₀={traj['x0']}, v₀={traj['v0']}), "
              f"Range: x∈[{traj['x'].min():.3f}, {traj['x'].max():.3f}], "
              f"v∈[{traj['v'].min():.3f}, {traj['v'].max():.3f}]")

if __name__ == "__main__":
    main()