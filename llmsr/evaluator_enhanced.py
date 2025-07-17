# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Enhanced Evaluator with Adaptive GPU/CPU Mode Selection

This module extends the original evaluator with automatic mode selection:
1. GPU mode for uncertainty-aware tasks (sequential execution)
2. CPU mode for standard tasks (multiprocessing)

Key improvements:
- Automatic detection of uncertainty tasks
- GPU memory management
- Backward compatibility with original interface
- Enhanced error handling and logging
"""

from __future__ import annotations
import pdb
from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type, Dict, Tuple, Optional, List
import profile
import multiprocessing
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from llmsr import code_manipulation
from llmsr import buffer
from llmsr import evaluator_accelerate
from tqdm import tqdm
import logging
import signal

logging.basicConfig(level=logging.ERROR, filename='evaluator.log')

class _FunctionLineVisitor(ast.NodeVisitor):
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name = target_function_name
        self._function_end_line = None
    def visit_FunctionDef(self, node: Any) -> None:
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)
    @property
    def function_end_line(self) -> int:
        assert self._function_end_line is not None
        return self._function_end_line

def _trim_function_body(generated_code: str) -> str:
    lines = generated_code.split('\n')
    tree = ast.parse(generated_code)
    visitor = _FunctionLineVisitor('equation')
    visitor.visit(tree)
    return '\n'.join(lines[:visitor.function_end_line])

class Evaluator(ABC):
    """Base evaluator class."""
    
    @abstractmethod
    def run(self, program: str, function_to_run: str, function_to_evolve: str, inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        pass

class AdaptiveEvaluator(Evaluator):
    """
    Enhanced evaluator that automatically switches between GPU and CPU modes.
    
    Features:
    - Automatic uncertainty task detection
    - GPU mode for uncertainty-aware tasks (no multiprocessing)
    - CPU mode for standard tasks (multiprocessing enabled)
    - Backward compatibility with original interface
    """
    
    def __init__(self, verbose: bool = False, timeout_seconds: int = 120):
        self._verbose = verbose
        self._timeout_seconds = timeout_seconds
        self._gpu_available = torch.cuda.is_available()
        self._force_cpu = False
        self._stats = {
            'gpu_mode_runs': 0,
            'cpu_mode_runs': 0,
            'total_runs': 0,
            'gpu_failures': 0,
            'cpu_failures': 0,
        }
        
        # Original evaluator compatibility
        self._numba_accelerate = False
        
        logging.info(f"AdaptiveEvaluator initialized - GPU available: {self._gpu_available}")
        
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, evaluate will run on CPU")
    
    def _detect_uncertainty_task(self, dataset: Dict[str, Any]) -> bool:
        """
        Detect if this is an uncertainty-aware task requiring GPU mode.
        
        Args:
            dataset: Input dataset
            
        Returns:
            True if uncertainty columns are detected
        """
        uncertainty_indicators = ['sigma_x', 'sigma_v', 'sigma_a', 'sigma_total']
        
        if isinstance(dataset, dict):
            dataset_keys = set(dataset.keys())
            logging.info(f"Dataset keys: {dataset_keys}")
            has_uncertainty = any(indicator in dataset_keys for indicator in uncertainty_indicators)
            
            if has_uncertainty:
                logging.info("Uncertainty task detected - using GPU sequential mode")
                return True
        
        logging.info("Standard task detected - using CPU multiprocessing mode")
        return False
    
    def _detect_multi_group_data(self, inputs: Any, test_input: str) -> bool:
        """
        Detect if this is multi-group data by checking for directory with group files.
        
        Args:
            inputs: Input data
            test_input: Test input key
            
        Returns:
            True if multi-group data is detected
        """
        if isinstance(inputs, dict) and test_input in inputs:
            data_path = inputs.get(f'{test_input}_path')
            if data_path and isinstance(data_path, (str, Path)):
                data_path = Path(data_path)
                if data_path.is_dir():
                    group_files = list(data_path.glob('group_*.csv'))
                    if len(group_files) > 1:
                        logging.info(f"Multi-group data detected: {len(group_files)} groups in {data_path}")
                        return True
        return False
    
    def _load_multi_group_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """
        Load multi-group data from directory containing group_*.csv files.
        
        Args:
            data_path: Path to directory containing group CSV files
            
        Returns:
            List of group datasets
        """
        group_files = sorted(data_path.glob('group_*.csv'))
        group_datasets = []
        
        for group_file in group_files:
            logging.info(f"Loading group data from {group_file}")
            
            try:
                df = pd.read_csv(group_file)
                
                # Convert to dataset format expected by evaluation functions
                group_data = {
                    't': df['t'].values,
                    'datax': df['datax'].values,
                    'datav': df['datav'].values,
                    'dataa': df['dataa'].values,
                    'sigma_x': df['sigma_x'].values,
                    'sigma_v': df['sigma_v'].values,
                    'sigma_a': df['sigma_a'].values,
                    'sigma_total': df['sigma_total'].values,
                }
                
                group_datasets.append(group_data)
                logging.info(f"Loaded group with {len(df)} points")
                
            except Exception as e:
                logging.error(f"Failed to load group data from {group_file}: {e}")
                continue
        
        return group_datasets
    
    def _convert_to_cpu_format(self, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert group data to CPU format (inputs/outputs structure).
        
        Args:
            group_data: Group data in GPU format
            
        Returns:
            Dataset in CPU format
        """
        # Extract time, position, velocity as inputs
        inputs = np.column_stack([
            group_data['t'],
            group_data['datax'],
            group_data['datav']
        ])
        
        # Acceleration as outputs
        outputs = group_data['dataa']
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            # Keep additional uncertainty data for reference
            'sigma_x': group_data.get('sigma_x'),
            'sigma_v': group_data.get('sigma_v'),
            'sigma_a': group_data.get('sigma_a'),
            'sigma_total': group_data.get('sigma_total')
        }
    
    def _run_multi_group_evaluation(self, program: str, function_to_run: str, 
                                   function_to_evolve: str, group_datasets: List[Dict[str, Any]],
                                   timeout_seconds: int, **kwargs) -> Tuple[Any, Any, bool]:
        """
        Run evaluation on multiple groups and aggregate results.
        
        Args:
            program: Program code to execute
            function_to_run: Function name to run
            function_to_evolve: Function name to evolve
            group_datasets: List of group datasets
            timeout_seconds: Timeout in seconds per group
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (aggregated_score, aggregated_params, success_flag)
        """
        logging.info(f"Running multi-group evaluation on {len(group_datasets)} groups")
        
        group_results = []
        successful_groups = 0
        
        for group_idx, group_data in enumerate(group_datasets):
            logging.info(f"Evaluating group {group_idx + 1}/{len(group_datasets)}")
            
            # Determine execution mode for this group
            use_gpu_mode = self._should_use_gpu_mode(group_data)
            logging.info(f"Group {group_idx + 1}: use_gpu_mode={use_gpu_mode}")
            
            if use_gpu_mode:
                result = self._run_gpu_mode(program, function_to_run, function_to_evolve,
                                          group_data, timeout_seconds, **kwargs)
            else:
                # Convert group_data to CPU format (inputs/outputs)
                cpu_dataset = self._convert_to_cpu_format(group_data)
                logging.info(f"Group {group_idx + 1}: converted to CPU format with keys: {cpu_dataset.keys()}")
                result = self._run_cpu_mode(program, function_to_run, function_to_evolve,
                                          cpu_dataset, timeout_seconds, **kwargs)
            
            score, params, runs_ok = result
            
            if runs_ok and score is not None:
                group_results.append({
                    'group_idx': group_idx,
                    'score': score,
                    'params': params,
                    'data_points': len(group_data['t'])
                })
                successful_groups += 1
                logging.info(f"Group {group_idx + 1} succeeded with score: {score:.6f}")
            else:
                logging.warning(f"Group {group_idx + 1} failed evaluation")
        
        if successful_groups == 0:
            logging.error("All groups failed evaluation")
            return (None, None, False)
        
        # Aggregate results from successful groups
        aggregated_score, aggregated_params = self._aggregate_multi_group_results(group_results)
        
        logging.info(f"Multi-group evaluation completed: {successful_groups}/{len(group_datasets)} groups succeeded")
        logging.info(f"Aggregated score: {aggregated_score:.6f}")
        
        return (aggregated_score, aggregated_params, True)
    
    def _aggregate_multi_group_results(self, group_results: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
        """
        Aggregate results from multiple groups.
        
        Args:
            group_results: List of group result dictionaries
            
        Returns:
            Tuple of (aggregated_score, aggregated_params)
        """
        if not group_results:
            return (0.0, [])
        
        # Calculate weighted average score (weighted by number of data points)
        total_weight = sum(result['data_points'] for result in group_results)
        weighted_score = sum(result['score'] * result['data_points'] for result in group_results) / total_weight
        
        # Average parameters across groups
        if group_results[0]['params']:
            n_params = len(group_results[0]['params'])
            aggregated_params = []
            
            for param_idx in range(n_params):
                param_values = [result['params'][param_idx] for result in group_results if result['params']]
                avg_param = sum(param_values) / len(param_values)
                aggregated_params.append(avg_param)
        else:
            aggregated_params = []
        
        logging.info(f"Aggregation: weighted score = {weighted_score:.6f}, "
                    f"total weight = {total_weight}, n_groups = {len(group_results)}")
        
        return (weighted_score, aggregated_params)
    
    def _should_use_gpu_mode(self, dataset: Dict[str, Any]) -> bool:
        """
        Determine execution mode based on task characteristics.
        
        Args:
            dataset: Input dataset
            
        Returns:
            True if GPU mode should be used
        """
        if self._force_cpu:
            return False
            
        if not self._gpu_available:
            return False
            
        return self._detect_uncertainty_task(dataset)
    
    def timeout_handler(self, signum, frame):
        """Original timeout handler for backward compatibility."""
        raise TimeoutError("Execution timed out")
    
    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
            inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Enhanced run method with adaptive mode selection and multi-group support.
        
        Args:
            program: Program code to execute
            function_to_run: Function name to run
            function_to_evolve: Function name to evolve
            inputs: Input data
            test_input: Test input key
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (results, success_flag)
        """
        self._stats['total_runs'] += 1
        
        # Check for multi-group data first
        if self._detect_multi_group_data(inputs, test_input):
            logging.info("Multi-group data detected, running multi-group evaluation")
            data_path = inputs.get(f'{test_input}_path')
            if data_path:
                group_datasets = self._load_multi_group_data(Path(data_path))
                if group_datasets:
                    results = self._run_multi_group_evaluation(
                        program, function_to_run, function_to_evolve,
                        group_datasets, timeout_seconds, **kwargs
                    )
                    
                    # Update statistics
                    if results[2]:  # runs_ok is True
                        self._stats['gpu_mode_runs'] += 1  # Multi-group typically uses GPU mode
                    else:
                        self._stats['gpu_failures'] += 1
                        
                    if self._verbose:
                        self._print_evaluation_details(program, results, **kwargs)
                        
                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    return results
                else:
                    logging.error("Failed to load multi-group data")
                    return (None, None, False)
            else:
                logging.error("Multi-group data path not provided")
                return (None, None, False)
        
        # Standard single-dataset evaluation
        dataset = inputs[test_input]
        
        # Determine execution mode
        use_gpu_mode = self._should_use_gpu_mode(dataset)
        
        if use_gpu_mode:
            self._stats['gpu_mode_runs'] += 1
            results = self._run_gpu_mode(program, function_to_run, function_to_evolve, 
                                       dataset, timeout_seconds, **kwargs)
        else:
            self._stats['cpu_mode_runs'] += 1
            # Convert to CPU format if needed
            if 'inputs' not in dataset:
                dataset = self._convert_to_cpu_format(dataset)
            results = self._run_cpu_mode(program, function_to_run, function_to_evolve, 
                                       dataset, timeout_seconds, **kwargs)
        
        # Update failure statistics
        if not results[2]:  # runs_ok is False
            if use_gpu_mode:
                self._stats['gpu_failures'] += 1
            else:
                self._stats['cpu_failures'] += 1
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def _run_gpu_mode(self, program: str, function_to_run: str, function_to_evolve: str,
                     dataset: Dict[str, Any], timeout_seconds: int, **kwargs) -> Tuple[Any, Any, bool]:
        """
        GPU mode: Sequential execution without multiprocessing.
        
        Used for uncertainty-aware tasks that require:
        - GPU computation
        - Automatic differentiation
        - Gradient computation
        """
        try:
            # Set up timeout signal handler
            def timeout_handler(signum, frame):
                raise TimeoutError("GPU execution timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                # Execute directly in main process (no multiprocessing)
                all_globals_namespace = {}
                exec(program, all_globals_namespace)
                function_to_run_obj = all_globals_namespace[function_to_run]
                
                logging.info(f"GPU mode: Executing {function_to_run_obj.__name__}")
                results = function_to_run_obj(dataset)
                
                # Validate results format
                if not isinstance(results, (tuple, list)) or len(results) != 2:
                    logging.error(f"Invalid evaluate output: expected tuple of (score, params), got {results}")
                    return (None, None, False)
                
                score, params = results
                
                # Validate result types
                if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                    logging.error(f"Invalid evaluate output types: score={type(score)}, params={type(params)}")
                    return (None, None, False)
                
                # Convert tensor to scalar if needed
                if isinstance(score, torch.Tensor):
                    score = score.item()
                
                signal.alarm(0)  # Cancel timeout
                return (score, params, True)
                
            except TimeoutError:
                logging.error(f"GPU execution timed out after {timeout_seconds} seconds")
                return (None, None, False)
            
        except Exception as e:
            logging.error(f"GPU mode execution error: {e}\\nProgram:\\n{program}")
            return (None, None, False)
        finally:
            signal.alarm(0)  # Ensure timeout is canceled
    
    def _run_cpu_mode(self, program: str, function_to_run: str, function_to_evolve: str,
                     dataset: Dict[str, Any], timeout_seconds: int, **kwargs) -> Tuple[Any, Any, bool]:
        """
        CPU mode: Multiprocessing execution for standard tasks.
        
        Used for standard tasks that:
        - Don't require GPU computation
        - Can benefit from multiprocessing parallelism
        - Don't need automatic differentiation
        """
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"Process timed out after {timeout_seconds} seconds")
            logging.error(f"Process timed out after {timeout_seconds} seconds")
            return (None, None, False)
        else:
            return self._get_results(result_queue)
    
    def _compile_and_run_function(self, program: str, function_to_run: str, 
                                 function_to_evolve: str, dataset: Dict[str, Any], 
                                 numba_accelerate: bool, result_queue: multiprocessing.Queue):
        """
        Compile and run function in subprocess (CPU mode only).
        
        This is the original multiprocessing execution logic.
        """
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run_obj = all_globals_namespace[function_to_run]
            
            logging.info(f"CPU mode: Executing {function_to_run_obj.__name__} with dataset shapes: inputs={dataset['inputs'].shape}, outputs={dataset['outputs'].shape}")
            results = function_to_run_obj(dataset)
            
            # Validate results format
            if not isinstance(results, (tuple, list)) or len(results) != 2:
                logging.error(f"Invalid evaluate output: expected tuple of (score, params), got {results}")
                result_queue.put((None, None, False))
                return
            
            score, params = results
            
            # Validate result types
            if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                logging.error(f"Invalid evaluate output types: score={type(score)}, params={type(params)}")
                result_queue.put((None, None, False))
                return
            
            # Convert tensor to scalar if needed
            if isinstance(score, torch.Tensor):
                score = score.item()
            
            result_queue.put((score, params, True))
            
        except Exception as e:
            logging.error(f"CPU mode execution error: {e}\\nProgram:\\n{program}\\nDataset shapes: inputs={dataset['inputs'].shape}, outputs={dataset['outputs'].shape}")
            result_queue.put((None, None, False))
    
    def _get_results(self, queue: multiprocessing.Queue) -> Tuple[Any, Any, bool]:
        """
        Get results from multiprocessing queue with retry logic.
        """
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return (None, None, False)
    
    def _print_evaluation_details(self, program: str, results: Tuple[Any, Any, bool], **kwargs):
        """
        Print evaluation details for debugging.
        """
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\\n-----------------------------------------------------')
        print(f'Results: {results}\\n=====================================================\\n\\n')
    
    def set_force_cpu(self, force: bool):
        """
        Force CPU mode regardless of task characteristics.
        
        Args:
            force: If True, always use CPU mode
        """
        self._force_cpu = force
        logging.info(f"Force CPU mode: {force}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        stats = self._stats.copy()
        stats.update({
            'gpu_available': self._gpu_available,
            'force_cpu': self._force_cpu,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'torch_version': torch.__version__,
        })
        return stats

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Check if program calls ancestor functions."""
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False

class BasicEvaluator(AdaptiveEvaluator):
    """
    Basic evaluator that extends AdaptiveEvaluator for program evaluation.
    
    This class maintains compatibility with the original evaluator interface
    while adding adaptive GPU/CPU mode selection.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._profile = None
        self._function_to_evolve = None
    
    def set_profile(self, profile_obj: Any):
        """Set profiling object."""
        self._profile = profile_obj
    
    def evaluate_programs(self, programs: Sequence[buffer.Function], inputs: dict, test_input: str, timeout_seconds: int, **kwargs) -> None:
        """
        Evaluate multiple programs and update their scores.
        
        Args:
            programs: Sequence of Function objects to evaluate
            inputs: Input data
            test_input: Test input key
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
        """
        self._function_to_evolve = kwargs.get('func_to_evolve', 'equation')
        
        for program in programs:
            if program.score is not None:
                continue  # Skip already evaluated programs
            
            # Run evaluation
            score, params, runs_ok = self.run(
                program.code, 'evaluate', self._function_to_evolve, 
                inputs, test_input, timeout_seconds, **kwargs
            )
            
            # Update program score
            if runs_ok and not _calls_ancestor(program.code, self._function_to_evolve) and score is not None:
                program.score = score
                
                # Update profiling if available
                if self._profile is not None:
                    self._profile.register_evaluated_program(program, score)
            else:
                # Set score to None for failed evaluations
                if self._profile is not None:
                    new_function = copy.deepcopy(program)
                    new_function.score = None
                    self._profile.register_evaluated_program(new_function, None)

# Factory function for backward compatibility
def create_evaluator(verbose: bool = False, **kwargs) -> BasicEvaluator:
    """
    Create evaluator instance with backward compatibility.
    
    Args:
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        BasicEvaluator instance
    """
    return BasicEvaluator(verbose=verbose)