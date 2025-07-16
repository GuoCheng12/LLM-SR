import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
train_path = '/fs-computility/ai4sData/wuguocheng/LLM-SR/data/oscillator_noise/oscillator2_train.csv'
test_ood_path = '//fs-computility/ai4sData/wuguocheng/LLM-SR/data/oscillator_noise/oscillator2_ood.csv'

# Load data
try:
    df_train = pd.read_csv(train_path)
    df_test_ood = pd.read_csv(test_ood_path)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit(1)

# Verify data structure
expected_columns = ['x', 'v', 'a']
for df, name in [(df_train, 'train'), (df_test_ood, 'test_ood')]:
    if not all(col in df.columns for col in expected_columns):
        print(f"Error: {name}.csv must contain columns: x, v, a")
        exit(1)

# Set seaborn style for aesthetics
sns.set(style='darkgrid')

# Phase Plot: x vs v
plt.figure(figsize=(10, 8))
# Training data
plt.plot(df_train['x'], df_train['v'], label='Train', color='blue', linewidth=2)
plt.scatter(df_train['x'], df_train['v'], s=10, color='blue', alpha=0.3)
# Test OOD data
plt.plot(df_test_ood['x'], df_test_ood['v'], label='Test OOD', color='orange', 
         linewidth=2, linestyle='--')
plt.scatter(df_test_ood['x'], df_test_ood['v'], s=10, color='orange', alpha=0.3)

# Labels and title
plt.xlabel('Displacement (x)', fontsize=12)
plt.ylabel('Velocity (v)', fontsize=12)
plt.title('Phase Plot: x vs v for Oscillator1 (Train, Test ID, Test OOD)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Save and display
plt.savefig('oscillator1_phase_plot_all.png', dpi=300)
plt.show()