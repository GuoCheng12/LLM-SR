#!/usr/bin/env python3
"""
Enhanced main.py with support for multi-group data evaluation.

This version supports:
1. Multi-group uncertainty data (osc2_noise)
2. Enhanced evaluator with GPU/CPU adaptive mode
3. Automatic detection of uncertainty tasks
4. Backward compatibility with original single-dataset experiments
"""

import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd
import pdb
from pathlib import Path
from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator_enhanced  # Use enhanced evaluator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str, required=True)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--data_path', type=str, default=None, help="Path to multi-group data directory")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--force_cpu', type=bool, default=False, help="Force CPU mode")
parser.add_argument('--timeout_seconds', type=int, default=300, help="Timeout per evaluation")

args = parser.parse_args()

def load_single_dataset(problem_name: str, use_torch: bool = False):
    """
    Load single dataset (original functionality).
    
    Args:
        problem_name: Problem name
        use_torch: Whether to convert to torch tensors
        
    Returns:
        Dataset dictionary
    """
    logging.info(f"Loading single dataset for problem: {problem_name}")
    
    df = pd.read_csv(f'./data/{problem_name}/train.csv')
    data = np.array(df)
    
    if "noise" in problem_name:
        # For noise problems, X = [t, x, v, sigma_x, sigma_v], y = [a, sigma_a]
        X = data[:, [0,1,2,4,5]]  # t, x, v, sigma_x, sigma_v
        y = data[:, [3, 6]]  # a and sigma_a
    else:
        # For non-noise problems, X = [t, x, v], y = a
        X = data[:, :3]  # t, x, v
        y = data[:, -1]  # a
    
    if use_torch:
        X = torch.Tensor(X)
        y = torch.Tensor(y)

    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}
    
    logging.info(f"Loaded single dataset: inputs shape {X.shape}, outputs shape {y.shape}")
    return dataset

def setup_multi_group_dataset(data_path: str, test_split: str = 'train'):
    """
    Set up multi-group dataset for enhanced evaluator.
    
    Args:
        data_path: Path to multi-group data directory
        test_split: Which split to use (train/test_id/test_ood)
        
    Returns:
        Dataset dictionary with path information
    """
    data_path = Path(data_path)
    
    # If data_path already includes the split (e.g., data/osc2_noise/train), use it directly
    if data_path.name == test_split:
        split_path = data_path
    else:
        split_path = data_path / test_split
    
    if not split_path.exists():
        raise FileNotFoundError(f"Multi-group data path not found: {split_path}")
    
    group_files = list(split_path.glob('group_*.csv'))
    if not group_files:
        raise FileNotFoundError(f"No group_*.csv files found in {split_path}")
    
    logging.info(f"Found {len(group_files)} groups in {split_path}")
    
    # Create dataset dict for multi-group evaluation
    dataset = {
        'data': None,  # Will be loaded by enhanced evaluator
        f'data_path': str(split_path)  # Path for multi-group loader
    }
    
    return dataset

def detect_uncertainty_task(spec_path: str) -> bool:
    """
    Detect if this is an uncertainty-aware task based on specification.
    
    Args:
        spec_path: Path to specification file
        
    Returns:
        True if uncertainty task detected
    """
    try:
        with open(spec_path, 'r') as f:
            spec_content = f.read()
        
        uncertainty_indicators = ['sigma_x', 'sigma_v', 'sigma_a', 'sigma_total', 'uncertainty']
        has_uncertainty = any(indicator in spec_content.lower() for indicator in uncertainty_indicators)
        
        if has_uncertainty:
            logging.info("Uncertainty task detected from specification")
        else:
            logging.info("Standard task detected from specification")
            
        return has_uncertainty
        
    except Exception as e:
        logging.warning(f"Could not analyze specification: {e}")
        return False

def setup_enhanced_evaluator(force_cpu: bool = False, timeout_seconds: int = 300):
    """
    Set up enhanced evaluator with appropriate configuration.
    
    Args:
        force_cpu: Force CPU mode
        timeout_seconds: Timeout per evaluation
        
    Returns:
        Enhanced evaluator instance
    """
    # Create enhanced evaluator
    evaluator = evaluator_enhanced.BasicEvaluator(verbose=True)
    
    # Also patch the pipeline to use enhanced evaluator
    import llmsr.pipeline as pipeline
    pipeline.evaluator = evaluator_enhanced  # Replace the module import
    
    # Create a wrapper function to ensure correct Evaluator instantiation
    def create_enhanced_evaluator(database, template, function_to_evolve, function_to_run, inputs, timeout_seconds=60, **kwargs):
        # Create enhanced evaluator with original parameters
        enhanced_eval = evaluator_enhanced.BasicEvaluator(verbose=True)
        # Set the original evaluator parameters
        enhanced_eval._database = database
        enhanced_eval._template = template
        enhanced_eval._function_to_evolve = function_to_evolve
        enhanced_eval._function_to_run = function_to_run
        enhanced_eval._inputs = inputs
        enhanced_eval._timeout_seconds = timeout_seconds
        
        # Force CPU mode for non-uncertainty tasks, but torch specs need GPU mode to avoid multiprocessing issues
        # Check if this is a torch specification based on global context
        import sys
        is_torch_spec = any('torch' in str(arg) for arg in sys.argv if 'spec' in str(arg))
        
        force_cpu_mode = force_cpu or (not is_uncertainty_task and not is_torch_spec)
        if force_cpu_mode:
            enhanced_eval.set_force_cpu(True)
            logging.info(f"Forced CPU mode: spec-based={not is_uncertainty_task}, user-forced={force_cpu}, torch-spec={is_torch_spec}")
        else:
            logging.info(f"Using GPU mode: uncertainty={is_uncertainty_task}, torch-spec={is_torch_spec}")
        
        return enhanced_eval
    
    # Replace the Evaluator class with our wrapper
    pipeline.evaluator.Evaluator = create_enhanced_evaluator
    
    # Set custom timeout
    evaluator._timeout_seconds = timeout_seconds
    
    return evaluator

def setup_class_config_with_enhanced_evaluator(evaluator):
    """
    Set up class config to use enhanced evaluator.
    
    Args:
        evaluator: Enhanced evaluator instance
        
    Returns:
        Modified class config
    """
    # Create a custom sandbox class that uses our enhanced evaluator
    class EnhancedSandbox:
        def __init__(self):
            self.evaluator = evaluator
            
        def run(self, *args, **kwargs):
            return self.evaluator.run(*args, **kwargs)
            
        def evaluate_programs(self, *args, **kwargs):
            return self.evaluator.evaluate_programs(*args, **kwargs)
    
    class_config = config.ClassConfig(
        llm_class=sampler.LocalLLM, 
        sandbox_class=EnhancedSandbox
    )
    
    return class_config

if __name__ == '__main__':
    print("Starting Enhanced LLM-SR Experiment")
    print("=" * 50)
    
    # Setup enhanced evaluator
    enhanced_evaluator = setup_enhanced_evaluator(
        force_cpu=args.force_cpu,
        timeout_seconds=args.timeout_seconds
    )
    
    # Load config and parameters
    experiment_config = config.Config(
        use_api=args.use_api, 
        api_model=args.api_model,
    )
    
    # Setup class config with enhanced evaluator
    class_config = setup_class_config_with_enhanced_evaluator(enhanced_evaluator)
    
    global_max_sample_num = 10000 
    
    # Load prompt specification
    logging.info(f"Loading specification from: {args.spec_path}")
    with open(args.spec_path, encoding="utf-8") as f:
        specification = f.read()
    
    # Detect task type
    is_uncertainty_task = detect_uncertainty_task(args.spec_path)
    
    # Setup dataset
    if args.data_path:
        # Multi-group data mode
        logging.info(f"Multi-group mode: using data from {args.data_path}")
        dataset = setup_multi_group_dataset(args.data_path, test_split='train')
        
        force_cpu_mode = args.force_cpu or not is_uncertainty_task
        print(f"Multi-group dataset configured:")
        print(f"  Data path: {args.data_path}")
        print(f"  Enhanced evaluator: {'CPU' if force_cpu_mode else 'GPU'} mode")
        print(f"  Uncertainty task: {is_uncertainty_task}")
        
    else:
        # Single dataset mode (backward compatibility)
        logging.info("Single dataset mode: using original data loading")
        use_torch = 'torch' in args.spec_path
        dataset = load_single_dataset(args.problem_name, use_torch=use_torch)
        
        print(f"Single dataset configured:")
        print(f"  Problem: {args.problem_name}")
        print(f"  Use torch: {use_torch}")
        print(f"  Enhanced evaluator: {'GPU' if not args.force_cpu else 'CPU'} mode")
    
    # Display configuration summary
    print(f"\nExperiment Configuration:")
    print(f"  API Model: {args.api_model if args.use_api else 'Local LLM'}")
    print(f"  Specification: {os.path.basename(args.spec_path)}")
    print(f"  Log Path: {args.log_path}")
    print(f"  Timeout: {args.timeout_seconds}s per evaluation")
    print(f"  Max Samples: {global_max_sample_num}")
    
    # Get evaluator stats
    stats = enhanced_evaluator.get_stats()
    print(f"\nEvaluator Status:")
    print(f"  GPU Available: {stats['gpu_available']}")
    print(f"  CUDA Version: {stats['cuda_version']}")
    print(f"  PyTorch Version: {stats['torch_version']}")
    
    print("\nStarting pipeline...")
    print("=" * 50)
    
    # Run pipeline
    try:
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=experiment_config,
            max_sample_nums=global_max_sample_num,
            class_config=class_config,
            log_dir=args.log_path,
        )
        
        # Print final evaluator statistics
        final_stats = enhanced_evaluator.get_stats()
        print("\nFinal Evaluator Statistics:")
        print("-" * 30)
        for key, value in final_stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\nExperiment completed successfully!")