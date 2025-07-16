import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# Oscillator2 equation: a = 0.3 sin(t) - 0.5 v^3 - x*v - 5 x exp(0.5 x)
def oscillator2(t, state):
    x, v = state
    a = 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)
    dx_dt = v
    dv_dt = a
    return [dx_dt, dv_dt]

def compute_sigma_a(x, v, sigma_x, sigma_v):
    term1 = (1.5* v**2 + x)**2 * sigma_v**2
    term2 = (v + 5 * np.exp(0.5* x) * (1 + 0.5 * x))**2 * sigma_x**2
    sigma_a = np.sqrt(term1 + term2)
    return sigma_a



# Parameters
sigma_std = 0.05 # standard deviation (gaussian)
t_start = 0.0
t_end = 50.0
points_per_file = 10000
total_points = 25000  # Oversample to ensure enough points for both segments
t_step = (t_end - t_start) / total_points  # Δt ≈ 0.002
t_eval = np.linspace(t_start, t_end, total_points)
initial_state = [0.5, 0.5]  # [x0, v0]

# Numerical integration
sol = solve_ivp(oscillator2, [t_start, t_end], initial_state, t_eval=t_eval, method='RK45')

# Extract data
t = t_eval
x = sol.y[0]
v = sol.y[1]
# Compute acceleration
a = 0.3 * np.sin(t) - 0.5 * v**3 - x * v - 5 * x * np.exp(0.5 * x)

# Create DataFrame
df = pd.DataFrame({
    't': t,
    'x': x,
    'v': v,
    'a': a
})

# Split into OOD and Train
# OOD: t ∈ [0, 20], 10,000 points
df_ood = df[df['t'] <= 20.0]
if len(df_ood) > points_per_file:
    df_ood = df_ood.iloc[::len(df_ood)//points_per_file][:points_per_file]
elif len(df_ood) < points_per_file:
    df_ood = df_ood.reindex(np.linspace(0, len(df_ood)-1, points_per_file)).interpolate()

# Train: t ∈ [30, 50], 10,000 points
df_train = df[df['t'] >= 30.0]
if len(df_train) > points_per_file:
    df_train = df_train.iloc[::len(df_train)//points_per_file][:points_per_file]
elif len(df_train) < points_per_file:
    df_train = df_train.reindex(np.linspace(0, len(df_train)-1, points_per_file)).interpolate()

# Add Gaussian noise to x and v in train set
np.random.seed(42)  # For reproducibility
# Sample sigma_x and sigma_v from N(0, 0.05^2), ensure non-negative
sigma_x = np.abs(np.random.normal(0, sigma_std, len(df_train)))
sigma_v = np.abs(np.random.normal(0, sigma_std, len(df_train)))
# Generate noise for x and v using sampled sigmas
x_noise = df_train['x'].values + np.random.normal(0, sigma_x, len(df_train))
v_noise = df_train['v'].values + np.random.normal(0, sigma_v, len(df_train))

a = df_train['a'].values
sigma_a = compute_sigma_a(x_noise, v_noise, sigma_x, sigma_v)

df_train = pd.DataFrame({
    't': df_train['t'].values,
    'x': x_noise,
    'v': v_noise,
    'a': a,
    'sigma_x': np.full(len(df_train), sigma_x),
    'sigma_v': np.full(len(df_train), sigma_v),
    'sigma_a': sigma_a
})

# Verify point counts
print(f"OOD data: {len(df_ood)} points, t in [{df_ood['t'].min()}, {df_ood['t'].max()}]")
print(f"Train data: {len(df_train)} points, t in [{df_train['t'].min()}, {df_train['t'].max()}]")
print("OOD head:\n", df_ood.head())
print("Train head:\n", df_train.head())

# Save to CSV
ood_path = '/fs-computility/ai4sData/wuguocheng/LLM-SR/data/oscillator_noise/oscillator2_ood.csv'
train_path = '/fs-computility/ai4sData/wuguocheng/LLM-SR/data/oscillator_noise/oscillator2_train.csv'
df_ood.to_csv(ood_path, index=False)
df_train.to_csv(train_path, index=False)
print(f"OOD data saved to {ood_path}")
print(f"Train data saved to {train_path}")