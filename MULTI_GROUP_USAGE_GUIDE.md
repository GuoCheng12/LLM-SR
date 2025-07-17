# Multi-Group Data Evaluation Usage Guide

## Overview

The enhanced evaluator now supports **multi-group data evaluation**, which automatically:
1. Detects when data consists of multiple groups (e.g., `group_1.csv`, `group_2.csv`, etc.)
2. Evaluates each group independently
3. Aggregates results using weighted averaging
4. Returns a single score representing performance across all groups

## Key Features

### 🎯 **Automatic Multi-Group Detection**
- Detects directory structure with `group_*.csv` files
- Automatically switches to multi-group evaluation mode
- Maintains backward compatibility with single-dataset evaluation

### 📊 **Robust Statistical Aggregation**
- **Weighted Average Score**: Groups with more data points contribute more to the final score
- **Parameter Averaging**: Parameters are averaged across successful groups
- **Failure Handling**: Evaluation succeeds if at least one group succeeds

### 🚀 **GPU-Optimized for Uncertainty Tasks**
- Automatically uses GPU mode for uncertainty-aware tasks
- Supports gradient computation for uncertainty propagation
- Efficient memory management across groups

## Usage Examples

### 1. Basic Multi-Group Evaluation

```python
from llmsr.evaluator_enhanced import BasicEvaluator

# Create evaluator
evaluator = BasicEvaluator(verbose=True)

# Set up inputs with multi-group data path
inputs = {
    'train': None,  # Will be auto-loaded from path
    'train_path': '/path/to/data/osc2_noise/train'  # Directory with group_*.csv files
}

# Run evaluation - automatically detects multi-group data
score, params, runs_ok = evaluator.run(
    program=program_code,
    function_to_run='evaluate', 
    function_to_evolve='equation',
    inputs=inputs,
    test_input='train',
    timeout_seconds=300
)

print(f"Multi-group score: {score}")
print(f"Aggregated parameters: {params}")
```

### 2. Using with Uncertainty Specification

```python
# For uncertainty-aware tasks with specification_oscillator2_uncertainty_torch.txt
# No changes needed - the evaluator automatically:
# 1. Detects uncertainty columns in each group
# 2. Uses GPU mode for all groups
# 3. Aggregates uncertainty-weighted results

# Example with your osc2_noise data
inputs = {
    'train': None,
    'train_path': '/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/train'  # 8 groups
}

# This will evaluate all 8 groups and return weighted average score
score, params, runs_ok = evaluator.run(
    program=uncertainty_program,
    function_to_run='evaluate',
    function_to_evolve='equation', 
    inputs=inputs,
    test_input='train',
    timeout_seconds=600  # More time for multiple groups
)
```

### 3. Integration with LLM-SR Pipeline

```python
# In your main experiment script
def run_experiment_with_multi_group():
    # Use enhanced evaluator in your pipeline
    from llmsr.evaluator_enhanced import BasicEvaluator
    
    evaluator = BasicEvaluator(verbose=True)
    
    # Set up data paths
    data_paths = {
        'train': '/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/train',
        'test_id': '/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/test_id', 
        'test_ood': '/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/test_ood'
    }
    
    inputs = {
        'train': None,
        'train_path': data_paths['train']
    }
    
    # Your existing pipeline code works unchanged
    # The evaluator automatically handles multi-group data
```

## Data Format Requirements

### Directory Structure
```
data/osc2_noise/train/
├── group_1.csv
├── group_2.csv
├── group_3.csv
├── ...
└── group_N.csv
```

### CSV Format (same for all groups)
```csv
t,datax,datav,dataa,sigma_x,sigma_v,sigma_a,sigma_total
30.000196,0.818416,0.834504,-7.194705,0.032966,0.095666,0.094170,0.422600
30.001320,-0.386690,-0.614526,1.216483,0.032966,0.095666,0.094170,0.130952
...
```

## Aggregation Strategy

### Score Aggregation
```python
# Weighted average by number of data points
total_weight = sum(group_data_points for each successful group)
final_score = sum(group_score * group_data_points) / total_weight
```

### Parameter Aggregation
```python
# Simple average across successful groups
final_params = [
    sum(group_params[i] for each group) / num_successful_groups
    for i in range(num_parameters)
]
```

## Performance Considerations

### GPU Memory Management
- Each group is evaluated sequentially to manage GPU memory
- Automatic GPU cache clearing between groups
- Memory usage scales with largest group, not total data

### Timeout Settings
- Timeout applies **per group**, not total evaluation
- Recommend longer timeouts for multi-group evaluation
- Failed groups don't affect successful ones

### Execution Statistics
```python
# Monitor performance
stats = evaluator.get_stats()
print(f"Total evaluations: {stats['total_runs']}")
print(f"GPU mode usage: {stats['gpu_mode_runs']}")
print(f"Success rate: {1 - stats['gpu_failures']/stats['total_runs']:.2%}")
```

## Example: Running Uncertainty Experiment

```bash
# Command line usage (modify your main.py to use enhanced evaluator)
python main.py --use_api True \
               --api_model "gpt-3.5-turbo" \
               --problem_name oscillator2_uncertainty \
               --spec_path ./specs/specification_oscillator2_uncertainty_torch.txt \
               --data_path ./data/osc2_noise/train \
               --log_path ./logs/osc2_uncertainty_multi_group_gpt3.5
```

## Migration from Single-Group

### Minimal Changes Required
```python
# Before (single group)
inputs = {
    'train': single_dataset  # Dictionary with data arrays
}

# After (multi-group)  
inputs = {
    'train': None,  # Auto-loaded
    'train_path': '/path/to/multi/group/data'  # Directory path
}

# Everything else remains the same!
```

### Backward Compatibility
- Single-group evaluation still works exactly as before
- No changes needed to existing evaluation functions
- Same API, enhanced functionality

## Troubleshooting

### Common Issues

1. **Path Not Found**
   ```python
   # Ensure data path exists and contains group_*.csv files
   data_path = Path('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/train')
   assert data_path.exists()
   assert len(list(data_path.glob('group_*.csv'))) > 0
   ```

2. **GPU Memory Issues**
   ```python
   # Reduce group size or force CPU mode
   evaluator.set_force_cpu(True)
   ```

3. **All Groups Failing**
   ```python
   # Check individual group evaluation
   # Enable verbose logging to see per-group results
   evaluator = BasicEvaluator(verbose=True)
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)

# Detailed execution logging
evaluator = BasicEvaluator(verbose=True)
```

## Expected Performance

### With 8 Groups (80,000 total points)
- **Evaluation Time**: 5-15 minutes depending on GPU and complexity
- **Memory Usage**: ~2-4GB GPU memory (depends on group size)
- **Success Rate**: Same as single-group evaluation for robust equations

### Score Characteristics
- **More Robust**: Averaged across multiple noise realizations
- **Better Generalization**: Represents performance across different noise conditions
- **Statistical Significance**: Based on larger, more diverse dataset

This multi-group evaluation provides more reliable and robust assessment of equation quality across different noise realizations and experimental conditions.