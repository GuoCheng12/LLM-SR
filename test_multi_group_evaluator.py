#!/usr/bin/env python3
"""
Test script for multi-group evaluator functionality.

This script demonstrates how to use the enhanced evaluator with multi-group data
from the osc2_noise dataset.
"""

import sys
import os
sys.path.append('/Users/wuguocheng/workshop/LLM-SR')

from llmsr.evaluator_enhanced import BasicEvaluator
import logging

# Configure logging to see detailed execution info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_multi_group_evaluation():
    """Test multi-group evaluation functionality."""
    
    print("Testing Multi-Group Evaluator")
    print("=" * 50)
    
    # Create evaluator
    evaluator = BasicEvaluator(verbose=True)
    
    # Test program using the uncertainty specification format
    test_program = '''
import torch
import numpy as np
import logging
from tqdm import tqdm

@evaluate.run
def evaluate(data: dict) -> tuple[float, list]:
    """Evaluate the equation on noisy data with dual supervision using GPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Running evaluate on {device}")

    t = torch.tensor(data['t'], device=device, requires_grad=False)
    x = torch.tensor(data['datax'], device=device, requires_grad=True)
    v = torch.tensor(data['datav'], device=device, requires_grad=True)
    a = torch.tensor(data['dataa'], device=device)
    sigma_x = torch.tensor(data['sigma_x'], device=device)
    sigma_v = torch.tensor(data['sigma_v'], device=device)
    sigma_a = torch.tensor(data['sigma_a'], device=device)
    sigma_total = torch.tensor(data['sigma_total'], device=device)

    LR = 1e-4
    N_ITERATIONS = 100  # Reduced for testing
    MAX_NPARAMS = 10
    PRAMS_INIT = [torch.nn.Parameter(torch.tensor(1.0, device=device)) for _ in range(MAX_NPARAMS)]

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.params = torch.nn.ParameterList(PRAMS_INIT)

        def forward(self, t, x, v):
            return equation(t, x, v, self.params)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(N_ITERATIONS):
        optimizer.zero_grad()
        y_pred = model(t, x, v)
        if torch.any(torch.isnan(y_pred)) or torch.any(torch.isinf(y_pred)):
            logging.error("NaN or Inf detected in y_pred")
            return None, None
        
        # Value loss with weighting by 1/sigma_total^2
        weights = 1.0 / (sigma_total ** 2 + 1e-8)
        value_loss = torch.mean(weights * (y_pred - a) ** 2)
        
        # Uncertainty loss
        grad_outputs = torch.ones_like(y_pred)
        try:
            gradients = torch.autograd.grad(y_pred, (x, v), grad_outputs=grad_outputs, create_graph=True)
            grad_x, grad_v = gradients[0], gradients[1]
            if torch.any(torch.isnan(grad_x)) or torch.any(torch.isnan(grad_v)):
                logging.error("NaN detected in gradients")
                return None, None
        except RuntimeError as e:
            logging.error(f"Gradient computation failed: {e}")
            return None, None
        
        pred_uncertainty = torch.sqrt((grad_x.abs() * sigma_x)**2 + (grad_v.abs() * sigma_v)**2 + 1e-8)
        expected_prop = torch.sqrt((sigma_total**2 - sigma_a**2).clamp(min=0.0))
        uncertainty_loss = torch.mean((pred_uncertainty - expected_prop) ** 2)
        
        loss = value_loss + uncertainty_loss
        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("NaN or Inf detected in loss")
            return None, None
        loss.backward()
        optimizer.step()

    if torch.isnan(loss) or torch.isinf(loss):
        return None, None
    params_values = [p.item() for p in model.params]
    return -loss.item(), params_values

@equation.evolve
def equation(t: torch.Tensor, x: torch.Tensor, v: torch.Tensor, params: torch.nn.ParameterList) -> torch.Tensor:
    """Mathematical function for acceleration in a damped nonlinear oscillator."""
    # Simple test equation
    dv = params[0] * torch.sin(t) + params[1] * x + params[2] * v + params[3]
    return dv
'''
    
    # Set up inputs for multi-group data
    inputs = {
        'train': None,  # Will be loaded from multi-group data
        'train_path': '/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/train'
    }
    
    print(f"Testing with data path: {inputs['train_path']}")
    
    # Run evaluation
    try:
        score, params, runs_ok = evaluator.run(
            program=test_program,
            function_to_run='evaluate',
            function_to_evolve='equation',
            inputs=inputs,
            test_input='train',
            timeout_seconds=300
        )
        
        print("\nEvaluation Results:")
        print("-" * 30)
        print(f"Success: {runs_ok}")
        print(f"Score: {score}")
        print(f"Parameters: {params}")
        
        # Print statistics
        stats = evaluator.get_stats()
        print(f"\nEvaluator Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def test_single_group_compatibility():
    """Test that single-group evaluation still works."""
    
    print("\nTesting Single-Group Compatibility")
    print("=" * 50)
    
    # Create evaluator
    evaluator = BasicEvaluator(verbose=True)
    
    # Load a single group for testing
    import pandas as pd
    df = pd.read_csv('/Users/wuguocheng/workshop/LLM-SR/data/osc2_noise/train/group_1.csv')
    
    # Convert to expected format
    single_group_data = {
        't': df['t'].values,
        'datax': df['datax'].values,
        'datav': df['datav'].values,
        'dataa': df['dataa'].values,
        'sigma_x': df['sigma_x'].values,
        'sigma_v': df['sigma_v'].values,
        'sigma_a': df['sigma_a'].values,
        'sigma_total': df['sigma_total'].values,
    }
    
    # Simple test program
    test_program = '''
@evaluate.run
def evaluate(data: dict) -> tuple[float, list]:
    """Simple test evaluate function."""
    import torch
    
    # Convert to tensors
    t = torch.tensor(data['t'])
    x = torch.tensor(data['datax'])
    
    # Simple computation
    result = torch.mean(t + x).item()
    params = [1.0, 2.0, 3.0]
    
    return result, params

@equation.evolve
def equation(t, x, v, params):
    """Simple test equation."""
    return params[0] * t + params[1] * x + params[2] * v
'''
    
    inputs = {
        'train': single_group_data
    }
    
    try:
        score, params, runs_ok = evaluator.run(
            program=test_program,
            function_to_run='evaluate',
            function_to_evolve='equation',
            inputs=inputs,
            test_input='train',
            timeout_seconds=60
        )
        
        print(f"Single-group Success: {runs_ok}")
        print(f"Single-group Score: {score}")
        print(f"Single-group Parameters: {params}")
        
    except Exception as e:
        print(f"Error in single-group test: {e}")

if __name__ == "__main__":
    test_multi_group_evaluation()
    test_single_group_compatibility()