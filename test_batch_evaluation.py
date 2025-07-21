#!/usr/bin/env python3
"""
Test script for batch GPU evaluation functionality.

This script tests the batch GPU evaluation feature by:
1. Creating sample programs for evaluation
2. Testing batch vs individual evaluation performance
3. Verifying results consistency
"""

import sys
import os
import time
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmsr import evaluator_enhanced
from llmsr import buffer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_dataset() -> Dict[str, Any]:
    """Create a test dataset for evaluation."""
    # Create synthetic oscillator data with uncertainty
    t = np.linspace(0, 10, 100)
    x = np.sin(t) + 0.1 * np.random.randn(len(t))
    v = np.cos(t) + 0.1 * np.random.randn(len(t))
    a = -np.sin(t) + 0.1 * np.random.randn(len(t))
    
    dataset = {
        't': t,
        'datax': x,
        'datav': v, 
        'dataa': a,
        'sigma_x': 0.1 * np.ones(len(t)),
        'sigma_v': 0.1 * np.ones(len(t)),
        'sigma_a': 0.1 * np.ones(len(t)),
        'sigma_total': 0.1 * np.ones(len(t))
    }
    
    return dataset

def create_test_programs() -> List[str]:
    """Create test programs for evaluation."""
    programs = [
        # Simple harmonic oscillator
        '''
def equation(t, datax, datav, sigma_x=None, sigma_v=None, sigma_a=None, sigma_total=None):
    omega = 1.0
    return -omega**2 * datax
''',
        # Damped harmonic oscillator
        '''
def equation(t, datax, datav, sigma_x=None, sigma_v=None, sigma_a=None, sigma_total=None):
    omega = 1.0
    gamma = 0.1
    return -omega**2 * datax - 2*gamma*datav
''',
        # Simple linear relationship
        '''
def equation(t, datax, datav, sigma_x=None, sigma_v=None, sigma_a=None, sigma_total=None):
    return -datax
''',
        # Velocity dependent
        '''
def equation(t, datax, datav, sigma_x=None, sigma_v=None, sigma_a=None, sigma_total=None):
    return -2.0 * datav
''',
    ]
    
    return programs

def create_evaluation_program(equation_code: str) -> str:
    """Create a complete evaluation program."""
    template = f'''
import torch
import numpy as np

{equation_code}

def evaluate(dataset):
    """Evaluate function for testing."""
    try:
        # Extract data
        t = dataset['t']
        datax = dataset['datax'] 
        datav = dataset['datav']
        dataa = dataset['dataa']
        
        # Get prediction from equation
        pred_a = equation(t, datax, datav)
        
        # Calculate loss (MSE)
        if isinstance(pred_a, torch.Tensor):
            loss = torch.mean((pred_a - torch.tensor(dataa, dtype=torch.float32))**2)
            score = -loss.item()  # Negative MSE as fitness
        else:
            loss = np.mean((pred_a - dataa)**2)
            score = -loss  # Negative MSE as fitness
            
        # Return score and dummy parameters
        params = [1.0, 0.1]  # dummy parameters
        
        return score, params
        
    except Exception as e:
        print(f"Evaluation error: {{e}}")
        return None, None
'''
    return template

def test_batch_evaluation():
    """Test batch GPU evaluation functionality."""
    print("Testing Batch GPU Evaluation")
    print("=" * 50)
    
    # Create test data
    dataset = create_test_dataset()
    programs = create_test_programs()
    
    print(f"Created test dataset with {len(dataset['t'])} points")
    print(f"Created {len(programs)} test programs")
    
    # Create evaluator with different batch sizes
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        print("-" * 30)
        
        # Create evaluator
        evaluator = evaluator_enhanced.AdaptiveEvaluator(
            verbose=False, 
            timeout_seconds=60, 
            batch_size=batch_size
        )
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for i, program_code in enumerate(programs):
            full_program = create_evaluation_program(program_code)
            task = {
                'program': full_program,
                'function_to_run': 'evaluate',
                'function_to_evolve': 'equation',
                'dataset': dataset,
                'timeout_seconds': 60,
                'task_id': f'test_task_{i}'
            }
            evaluation_tasks.append(task)
        
        # Test batch evaluation
        start_time = time.time()
        results = evaluator.evaluate_batch_gpu(evaluation_tasks)
        batch_time = time.time() - start_time
        
        # Check results
        successful_results = [r for r in results if r[2]]  # r[2] is success flag
        
        print(f"Batch evaluation completed in {batch_time:.3f}s")
        print(f"Successful evaluations: {len(successful_results)}/{len(programs)}")
        
        for i, (score, params, success) in enumerate(results):
            if success:
                print(f"  Program {i+1}: score={score:.6f}, params={params}")
            else:
                print(f"  Program {i+1}: FAILED")
        
        # Print statistics
        stats = evaluator.get_stats()
        print(f"Evaluator stats: {stats}")

def test_individual_vs_batch_performance():
    """Compare individual vs batch evaluation performance."""
    print("\nComparing Individual vs Batch Performance")
    print("=" * 50)
    
    dataset = create_test_dataset()
    programs = create_test_programs()
    
    # Test individual evaluation
    evaluator_individual = evaluator_enhanced.AdaptiveEvaluator(verbose=False, batch_size=1)
    
    start_time = time.time()
    individual_results = []
    for program_code in programs:
        full_program = create_evaluation_program(program_code)
        result = evaluator_individual._run_single_gpu_task(
            full_program, 'evaluate', 'equation', dataset, 60
        )
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    # Test batch evaluation
    evaluator_batch = evaluator_enhanced.AdaptiveEvaluator(verbose=False, batch_size=4)
    
    evaluation_tasks = []
    for i, program_code in enumerate(programs):
        full_program = create_evaluation_program(program_code)
        task = {
            'program': full_program,
            'function_to_run': 'evaluate',
            'function_to_evolve': 'equation',
            'dataset': dataset,
            'timeout_seconds': 60,
            'task_id': f'perf_test_{i}'
        }
        evaluation_tasks.append(task)
    
    start_time = time.time()
    batch_results = evaluator_batch.evaluate_batch_gpu(evaluation_tasks)
    batch_time = time.time() - start_time
    
    print(f"Individual evaluation time: {individual_time:.3f}s")
    print(f"Batch evaluation time: {batch_time:.3f}s")
    print(f"Speedup ratio: {individual_time/batch_time:.2f}x")
    
    # Verify results consistency
    print("\nResults consistency check:")
    for i, (ind_result, batch_result) in enumerate(zip(individual_results, batch_results)):
        if ind_result[2] and batch_result[2]:  # Both successful
            score_diff = abs(ind_result[0] - batch_result[0])
            print(f"  Program {i+1}: score difference = {score_diff:.6f}")
        else:
            print(f"  Program {i+1}: consistency check failed (one or both failed)")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires GPU support.")
        sys.exit(1)
    
    try:
        test_batch_evaluation()
        test_individual_vs_batch_performance()
        print("\nBatch evaluation tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)