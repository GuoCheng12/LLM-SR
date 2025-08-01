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
    - Batch GPU evaluation for improved efficiency
    - Backward compatibility with original interface
    """
    
    def __init__(self, verbose: bool = False, timeout_seconds: int = 120, batch_size: int = 4):
        self._verbose = verbose
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._gpu_available = torch.cuda.is_available()
        self._force_cpu = False
        self._stats = {
            'gpu_mode_runs': 0,
            'cpu_mode_runs': 0,
            'total_runs': 0,
            'gpu_failures': 0,
            'cpu_failures': 0,
            'batch_runs': 0,
            'batch_efficiency': 0.0,
        }
        
        # Batch processing queue
        self._batch_queue = []
        self._batch_results = {}
        
        # Original evaluator compatibility
        self._numba_accelerate = False
        
        logging.info(f"AdaptiveEvaluator initialized - GPU available: {self._gpu_available}, batch_size: {self._batch_size}")
        
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, evaluate will run on CPU")
    
    def _detect_uncertainty_task(self, dataset: Dict[str, Any]) -> bool:
        """
        Detect if this is an uncertainty-aware task or torch-based task requiring GPU mode.
        
        Args:
            dataset: Input dataset
            
        Returns:
            True if uncertainty columns are detected or torch tensors are used
        """
        uncertainty_indicators = ['sigma_x', 'sigma_v', 'sigma_a', 'sigma_total']
        
        if isinstance(dataset, dict):
            dataset_keys = set(dataset.keys())
            logging.info(f"Dataset keys: {dataset_keys}")
            
            # Check for uncertainty indicators
            has_uncertainty = any(indicator in dataset_keys for indicator in uncertainty_indicators)
            
            # Check for torch tensors (which indicate torch-based specification)
            has_torch_tensors = any(isinstance(v, torch.Tensor) for v in dataset.values())
            
            if has_uncertainty:
                logging.info("Uncertainty task detected - using GPU sequential mode")
                return True
            elif has_torch_tensors:
                logging.info("Torch tensor data detected - using GPU sequential mode")
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
            'inputs': inputs,  # Keep as numpy array for CPU mode
            'outputs': outputs,  # Keep as numpy array for CPU mode
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
    
    def evaluate_batch_gpu(self, evaluation_tasks: List[Dict[str, Any]], **kwargs) -> List[Tuple[Any, Any, bool]]:
        """
        Batch GPU evaluation for improved efficiency.
        
        Args:
            evaluation_tasks: List of evaluation task dictionaries containing:
                - program: Program code to execute
                - function_to_run: Function name to run
                - function_to_evolve: Function name to evolve
                - dataset: Input dataset
                - timeout_seconds: Timeout in seconds
                - task_id: Unique identifier for the task
            **kwargs: Additional arguments
            
        Returns:
            List of tuples (score, params, success_flag) corresponding to each task
        """
        if not evaluation_tasks:
            return []
            
        self._stats['batch_runs'] += 1
        batch_start_time = time.time()
        
        logging.info(f"Starting batch GPU evaluation of {len(evaluation_tasks)} tasks")
        
        results = []
        successful_tasks = 0
        
        try:
            # Process all tasks in the batch sequentially but with optimized GPU memory management
            with torch.cuda.device(0) if torch.cuda.is_available() else torch.no_grad():
                for i, task in enumerate(evaluation_tasks):
                    logging.info(f"Processing batch task {i+1}/{len(evaluation_tasks)}")
                    
                    try:
                        result = self._run_single_gpu_task(
                            task['program'], 
                            task['function_to_run'],
                            task['function_to_evolve'],
                            task['dataset'],
                            task.get('timeout_seconds', self._timeout_seconds),
                            **kwargs
                        )
                        
                        if result[2]:  # success
                            successful_tasks += 1
                            
                        results.append(result)
                        
                        # Clear intermediate GPU memory after each task
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logging.error(f"Batch task {i+1} failed: {e}")
                        results.append((None, None, False))
                        
        finally:
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        batch_time = time.time() - batch_start_time
        efficiency = successful_tasks / len(evaluation_tasks) if evaluation_tasks else 0.0
        self._stats['batch_efficiency'] = (self._stats['batch_efficiency'] * (self._stats['batch_runs'] - 1) + efficiency) / self._stats['batch_runs']
        
        logging.info(f"Batch evaluation completed: {successful_tasks}/{len(evaluation_tasks)} tasks succeeded in {batch_time:.2f}s")
        
        return results
    
    def _run_single_gpu_task(self, program: str, function_to_run: str, function_to_evolve: str,
                            dataset: Dict[str, Any], timeout_seconds: int, **kwargs) -> Tuple[Any, Any, bool]:
        """
        Run a single GPU task with optimized memory management.
        
        Args:
            program: Program code to execute
            function_to_run: Function name to run
            function_to_evolve: Function name to evolve
            dataset: Input dataset
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (score, params, success_flag)
        """
        try:
            # Set up timeout signal handler
            def timeout_handler(signum, frame):
                raise TimeoutError("GPU task execution timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                # Execute directly in main process (no multiprocessing)
                all_globals_namespace = {}
                exec(program, all_globals_namespace)
                function_to_run_obj = all_globals_namespace[function_to_run]
                
                logging.info(f"GPU task: Executing {function_to_run_obj.__name__}")
                logging.info(f"GPU task: Dataset keys: {list(dataset.keys())}")
                logging.info(f"GPU task: Dataset types: {[(k, type(v)) for k, v in dataset.items()]}")
                
                results = function_to_run_obj(dataset)
                logging.info(f"GPU task: Function returned: {type(results)}, value: {results}")
                
                # Validate results format
                if not isinstance(results, (tuple, list)) or len(results) != 2:
                    logging.error(f"GPU task: Invalid evaluate output: expected tuple of (score, params), got {results}")
                    return (None, None, False)
                
                score, params = results
                logging.info(f"GPU task: Extracted score={score} (type: {type(score)}), params={params} (type: {type(params)})")
                
                # Handle None results from function execution errors
                if score is None or params is None:
                    logging.error(f"GPU task: Function returned None values - score={score}, params={params}")
                    return (None, None, False)
                
                # Validate result types
                if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                    logging.error(f"GPU task: Invalid evaluate output types: score={type(score)}, params={type(params)}")
                    return (None, None, False)
                
                # Convert tensor to scalar if needed
                if isinstance(score, torch.Tensor):
                    score = score.item()
                
                logging.info(f"GPU task: Final result - score={score}, params={params}")
                signal.alarm(0)  # Cancel timeout
                return (score, params, True)
                
            except TimeoutError:
                logging.error(f"GPU task execution timed out after {timeout_seconds} seconds")
                return (None, None, False)
            
        except Exception as e:
            logging.error(f"GPU task execution error: {e}")
            import traceback
            logging.error(f"GPU task traceback: {traceback.format_exc()}")
            return (None, None, False)
        finally:
            signal.alarm(0)  # Ensure timeout is canceled
    
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
            logging.info(f"Data path: {data_path}")
            if data_path:
                group_datasets = self._load_multi_group_data(Path(data_path))
                logging.info(f"Loaded {len(group_datasets) if group_datasets else 0} groups")
                if group_datasets:
                    logging.info(f"First group keys: {list(group_datasets[0].keys()) if group_datasets else 'None'}")
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
            logging.info(f"CPU mode: Starting evaluation with dataset keys: {list(dataset.keys())}")
            logging.info(f"CPU mode: Dataset types: {[(k, type(v)) for k, v in dataset.items()]}")
            
            all_globals_namespace = {}
            logging.info(f"CPU mode: Executing program...")
            exec(program, all_globals_namespace)
            
            if function_to_run not in all_globals_namespace:
                logging.error(f"CPU mode: Function '{function_to_run}' not found in program namespace")
                result_queue.put((None, None, False))
                return
                
            function_to_run_obj = all_globals_namespace[function_to_run]
            logging.info(f"CPU mode: Found function {function_to_run_obj.__name__}")
            
            if 'inputs' in dataset and 'outputs' in dataset:
                logging.info(f"CPU mode: Dataset shapes - inputs={dataset['inputs'].shape}, outputs={dataset['outputs'].shape}")
            else:
                logging.info(f"CPU mode: Dataset in direct format with keys: {list(dataset.keys())}")
            
            logging.info(f"CPU mode: Calling {function_to_run_obj.__name__}...")
            results = function_to_run_obj(dataset)
            logging.info(f"CPU mode: Function returned: {type(results)}, value: {results}")
            
            # Validate results format
            if not isinstance(results, (tuple, list)) or len(results) != 2:
                logging.error(f"CPU mode: Invalid evaluate output: expected tuple of (score, params), got {results}")
                result_queue.put((None, None, False))
                return
            
            score, params = results
            logging.info(f"CPU mode: Extracted score={score} (type: {type(score)}), params={params} (type: {type(params)})")
            
            # Validate result types
            if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                logging.error(f"CPU mode: Invalid evaluate output types: score={type(score)}, params={type(params)}")
                result_queue.put((None, None, False))
                return
            
            # Convert tensor to scalar if needed
            if isinstance(score, torch.Tensor):
                score = score.item()
            
            logging.info(f"CPU mode: Final result - score={score}, params={params}")
            result_queue.put((score, params, True))
            
        except Exception as e:
            try:
                input_shape = dataset['inputs'].shape if 'inputs' in dataset else 'No inputs'
                output_shape = dataset['outputs'].shape if 'outputs' in dataset else 'No outputs'
                logging.error(f"CPU mode execution error: {e}\\nProgram:\\n{program}\\nDataset shapes: inputs={input_shape}, outputs={output_shape}")
            except:
                logging.error(f"CPU mode execution error: {e}\\nProgram:\\n{program}\\nDataset type: {type(dataset)}")
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
        try:
            function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
            print(f'{str(function).strip()}\\n-----------------------------------------------------')
        except Exception as e:
            logging.warning(f"Could not parse program for display: {e}")
            print(f'[Program parsing failed: {e}]\\n-----------------------------------------------------')
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
        self._database = None
        self._template = None
        self._inputs = None
        self._timeout_seconds = 60
    
    def set_profile(self, profile_obj: Any):
        """Set profiling object."""
        self._profile = profile_obj
    
    def analyse(self, sample: str, island_id: int | None, version_generated: int | None, **kwargs) -> None:
        """
        Analyze a sample and register results in the database.
        
        Args:
            sample: Generated code sample
            island_id: Island identifier
            version_generated: Version number
            **kwargs: Additional arguments
        """
        if not self._database or not self._template:
            logging.error("Database or template not initialized for analyse method")
            return
            
        new_function, program = self._sample_to_program(sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        params_per_test = {}
        time_reset = time.time()

        for current_input in self._inputs:
            # Skip path keys in multi-group scenario
            if current_input.endswith('_path'):
                continue
                
            score, params, runs_ok = self.run(
                program, 'evaluate', self._function_to_evolve, self._inputs, current_input, self._timeout_seconds
            )

            logging.info(f"Run result: score={score}, params={params}, runs_ok={runs_ok}")
            if runs_ok and not self._calls_ancestor(program, self._function_to_evolve) and score is not None:
                if not isinstance(score, (int, float, torch.Tensor, np.floating)):
                    logging.error(f"Invalid evaluate output type: {type(score)}")
                    raise ValueError('@function.run did not return a numeric score.')
                if isinstance(score, torch.Tensor):
                    score = score.item()
                scores_per_test[current_input] = score
                params_per_test[current_input] = params
        
        evaluate_time = time.time() - time_reset
        if scores_per_test:
            self._database.register_program(new_function, island_id, scores_per_test, params_per_test=params_per_test, evaluate_time=evaluate_time, **kwargs)
        else:
            profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.params = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)
    
    def _sample_to_program(self, generated_code: str, version_generated: int | None, template, function_to_evolve: str):
        """Convert sample to program."""
        body = self._trim_function_body(generated_code)
        if version_generated is not None:
            import llmsr.code_manipulation as code_manipulation
            body = code_manipulation.rename_function_calls(
                code=body,
                source_name=f'{function_to_evolve}_v{version_generated}',
                target_name=function_to_evolve
            )
        program = copy.deepcopy(template)
        evolved_function = program.get_function(function_to_evolve)
        evolved_function.body = body
        return evolved_function, str(program)
    
    def _trim_function_body(self, code: str) -> str:
        """Trim function body from generated code."""
        if not code:
            return ''
        code = f'def fake_function_header():\n{code}'
        tree = None
        while tree is None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                code = code[:-1]
                if not code:
                    return ''
        visitor = _FunctionLineVisitor('fake_function_header')
        visitor.visit(tree)
        
        # Check if function was found
        if visitor._function_end_line is None:
            logging.warning(f"Could not find function end line in generated code: {code}")
            # Return the original code without the fake header
            return '\n'.join(code.splitlines()[1:]) + '\n\n'
            
        body_lines = code.splitlines()[1:visitor.function_end_line]
        return '\n'.join(body_lines) + '\n\n'
    
    def _calls_ancestor(self, program: str, function_to_evolve: str) -> bool:
        """Check if program calls ancestor functions."""
        import llmsr.code_manipulation as code_manipulation
        for name in code_manipulation.get_functions_called(program):
            if name.startswith(f'{function_to_evolve}_v'):
                return True
        return False
    
    def evaluate_programs(self, programs: Sequence[buffer.Function], inputs: dict, test_input: str, timeout_seconds: int, **kwargs) -> None:
        """
        Evaluate multiple programs and update their scores with batch optimization.
        
        Args:
            programs: Sequence of Function objects to evaluate
            inputs: Input data
            test_input: Test input key
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
        """
        self._function_to_evolve = kwargs.get('func_to_evolve', 'equation')
        dataset = inputs[test_input]
        
        # Collect unevaluated programs
        unevaluated_programs = [p for p in programs if p.score is None]
        
        if not unevaluated_programs:
            logging.info("All programs already evaluated, skipping")
            return
        
        # Check if we should use batch GPU evaluation
        use_gpu_mode = self._should_use_gpu_mode(dataset)
        
        if use_gpu_mode and len(unevaluated_programs) >= 2:
            # Use batch GPU evaluation for efficiency
            logging.info(f"Using batch GPU evaluation for {len(unevaluated_programs)} programs")
            self._evaluate_programs_batch_gpu(unevaluated_programs, inputs, test_input, timeout_seconds, **kwargs)
        else:
            # Fall back to individual evaluation
            logging.info(f"Using individual evaluation for {len(unevaluated_programs)} programs")
            self._evaluate_programs_individual(unevaluated_programs, inputs, test_input, timeout_seconds, **kwargs)
    
    def _evaluate_programs_batch_gpu(self, programs: List[buffer.Function], inputs: dict, test_input: str, timeout_seconds: int, **kwargs) -> None:
        """
        Evaluate programs using batch GPU processing.
        
        Args:
            programs: List of Function objects to evaluate
            inputs: Input data
            test_input: Test input key
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
        """
        dataset = inputs[test_input]
        
        # Process programs in batches
        for batch_start in range(0, len(programs), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(programs))
            batch_programs = programs[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_start//self._batch_size + 1}: programs {batch_start+1}-{batch_end}")
            
            # Prepare evaluation tasks for this batch
            evaluation_tasks = []
            for program in batch_programs:
                task = {
                    'program': program.code,
                    'function_to_run': 'evaluate',
                    'function_to_evolve': self._function_to_evolve,
                    'dataset': dataset,
                    'timeout_seconds': timeout_seconds,
                    'program_obj': program  # Keep reference to original program object
                }
                evaluation_tasks.append(task)
            
            # Run batch evaluation
            results = self.evaluate_batch_gpu(evaluation_tasks, **kwargs)
            
            # Update program scores with results
            for task, result in zip(evaluation_tasks, results):
                program = task['program_obj']
                score, params, runs_ok = result
                
                if runs_ok and not self._calls_ancestor(program.code, self._function_to_evolve) and score is not None:
                    program.score = score
                    
                    # Update profiling if available
                    if self._profile is not None:
                        self._profile.register_evaluated_program(program, score)
                else:
                    # Set score to None for failed evaluations
                    if self._profile is not None:
                        failed_program = copy.deepcopy(program)
                        failed_program.score = None
                        self._profile.register_evaluated_program(failed_program, None)
    
    def _evaluate_programs_individual(self, programs: List[buffer.Function], inputs: dict, test_input: str, timeout_seconds: int, **kwargs) -> None:
        """
        Evaluate programs individually (fallback method).
        
        Args:
            programs: List of Function objects to evaluate
            inputs: Input data
            test_input: Test input key
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments
        """
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