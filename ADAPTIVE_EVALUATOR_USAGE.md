# Adaptive Evaluator Usage Guide

## Overview

The Adaptive Evaluator framework automatically switches between GPU and CPU execution modes based on task characteristics:

- **GPU Mode**: Sequential execution for uncertainty-aware tasks
- **CPU Mode**: Multiprocessing execution for standard tasks

## Key Features

### 🔍 **Automatic Task Detection**
- Detects uncertainty columns (`sigma_x`, `sigma_v`, `sigma_a`, `sigma_total`)
- Switches to GPU mode for uncertainty-aware tasks
- Uses CPU multiprocessing for standard tasks

### 🚀 **GPU Mode Benefits**
- Supports automatic differentiation
- Handles gradient computation for uncertainty propagation
- Avoids CUDA multiprocessing conflicts
- Efficient memory management

### ⚡ **CPU Mode Benefits**
- Parallel execution with multiprocessing
- Better for standard symbolic regression tasks
- Lower memory usage
- Traditional timeout handling

## Usage

### 1. Basic Usage (Drop-in Replacement)

```python
from llmsr.evaluator_enhanced import create_evaluator

# Create adaptive evaluator
evaluator = create_evaluator(verbose=True)

# Use exactly like original evaluator
score, params, runs_ok = evaluator.run(
    program=program_code,
    function_to_run='evaluate',
    function_to_evolve='equation',
    inputs=inputs,
    test_input='train',
    timeout_seconds=120
)
```

### 2. Advanced Usage with Mode Control

```python
from llmsr.evaluator_enhanced import BasicEvaluator

# Create evaluator with custom settings
evaluator = BasicEvaluator(verbose=True)

# Force CPU mode (disable GPU detection)
evaluator.set_force_cpu(True)

# Evaluate programs
evaluator.evaluate_programs(
    programs=program_list,
    inputs=inputs,
    test_input='train',
    timeout_seconds=120
)

# Get execution statistics
stats = evaluator.get_stats()
print(f"GPU runs: {stats['gpu_mode_runs']}")
print(f"CPU runs: {stats['cpu_mode_runs']}")
```

### 3. Using with Uncertainty-Aware Specs

```python
# For uncertainty tasks like oscillator2_uncertainty_torch.txt
# No changes needed - automatic detection will use GPU mode

# Example dataset with uncertainty columns
dataset = {
    't': time_data,
    'datax': position_data,
    'datav': velocity_data,
    'dataa': acceleration_data,
    'sigma_x': position_uncertainty,
    'sigma_v': velocity_uncertainty,
    'sigma_a': acceleration_uncertainty,
    'sigma_total': total_uncertainty
}
```

## Implementation Details

### Mode Selection Logic

```python
def _should_use_gpu_mode(self, dataset):
    """
    GPU mode is used when:
    1. GPU is available
    2. Uncertainty columns are detected
    3. Force CPU is not enabled
    """
    if self._force_cpu:
        return False
    if not self._gpu_available:
        return False
    return self._detect_uncertainty_task(dataset)
```

### Uncertainty Detection

```python
def _detect_uncertainty_task(self, dataset):
    """
    Detects uncertainty columns:
    - sigma_x, sigma_v, sigma_a, sigma_total
    """
    uncertainty_indicators = ['sigma_x', 'sigma_v', 'sigma_a', 'sigma_total']
    dataset_keys = set(dataset.keys())
    return any(indicator in dataset_keys for indicator in uncertainty_indicators)
```

## Migration Guide

### From Original Evaluator

```python
# Before
from llmsr.evaluator import BasicEvaluator
evaluator = BasicEvaluator(verbose=True)

# After
from llmsr.evaluator_enhanced import BasicEvaluator
evaluator = BasicEvaluator(verbose=True)
# Everything else remains the same!
```

### Configuration Options

```python
# Create evaluator with custom timeout
evaluator = BasicEvaluator(verbose=True)
evaluator._timeout_seconds = 300  # 5 minutes

# Force CPU mode for debugging
evaluator.set_force_cpu(True)

# Check GPU availability
stats = evaluator.get_stats()
print(f"GPU available: {stats['gpu_available']}")
print(f"CUDA version: {stats['cuda_version']}")
```

## Performance Considerations

### GPU Mode
- **Pros**: Automatic differentiation, gradient computation, no multiprocessing overhead
- **Cons**: Sequential execution, higher memory usage
- **Best for**: Uncertainty-aware tasks, gradient-based optimization

### CPU Mode
- **Pros**: Parallel execution, lower memory usage, traditional timeout handling
- **Cons**: No automatic differentiation, multiprocessing overhead
- **Best for**: Standard symbolic regression, parameter optimization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Force CPU mode
   evaluator.set_force_cpu(True)
   ```

2. **Multiprocessing Issues**
   ```python
   # GPU mode automatically avoids multiprocessing
   # Check if uncertainty detection is working
   stats = evaluator.get_stats()
   print(f"GPU mode runs: {stats['gpu_mode_runs']}")
   ```

3. **Performance Issues**
   ```python
   # Check execution statistics
   stats = evaluator.get_stats()
   print(f"GPU failures: {stats['gpu_failures']}")
   print(f"CPU failures: {stats['cpu_failures']}")
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable verbose output
evaluator = BasicEvaluator(verbose=True)
```

## Example: Using with New Uncertainty Specs

```python
# Run experiment with uncertainty-aware specification
python main.py --use_api True \
               --api_model "gpt-3.5-turbo" \
               --problem_name oscillator2_uncertainty \
               --spec_path ./specs/specification_oscillator2_uncertainty_torch.txt \
               --log_path ./logs/osc2_uncertainty_gpt3.5

# The evaluator will automatically:
# 1. Detect uncertainty columns in dataset
# 2. Switch to GPU mode
# 3. Use sequential execution (no multiprocessing)
# 4. Support automatic differentiation
```

## Statistics and Monitoring

```python
# Get detailed execution statistics
stats = evaluator.get_stats()
print(f"""
Execution Statistics:
- Total runs: {stats['total_runs']}
- GPU mode runs: {stats['gpu_mode_runs']}
- CPU mode runs: {stats['cpu_mode_runs']}
- GPU failures: {stats['gpu_failures']}
- CPU failures: {stats['cpu_failures']}
- GPU available: {stats['gpu_available']}
- Force CPU: {stats['force_cpu']}
- CUDA version: {stats['cuda_version']}
- PyTorch version: {stats['torch_version']}
""")
```

This framework provides seamless integration with your existing codebase while adding intelligent mode selection for optimal performance.