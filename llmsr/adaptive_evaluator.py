"""
Adaptive Evaluator Framework for LLM-SR

This module provides an adaptive evaluation framework that automatically switches between:
1. GPU mode (sequential execution) for uncertainty-aware tasks
2. CPU mode (multiprocessing) for standard tasks

Design principles:
- Detect uncertainty columns in dataset to determine execution mode
- GPU mode: Sequential execution to avoid CUDA multiprocessing issues
- CPU mode: Multiprocessing for parallel evaluation
- Maintain compatibility with existing codebase
"""

import multiprocessing
import signal
import time
import logging
import torch
import numpy as np
from typing import Any, Tuple, Dict, List, Optional
from . import code_manipulation

class AdaptiveEvaluator:
    """
    Adaptive evaluator that switches between GPU and CPU modes based on task requirements.
    """
    
    def __init__(self, verbose: bool = False, timeout_seconds: int = 120):
        self._verbose = verbose
        self._timeout_seconds = timeout_seconds
        self._gpu_available = torch.cuda.is_available()
        self._force_cpu = False  # Override flag
        
        logging.info(f"AdaptiveEvaluator initialized - GPU available: {self._gpu_available}")
        
    def _detect_uncertainty_task(self, dataset: Dict[str, Any]) -> bool:
        """
        Detect if this is an uncertainty-aware task by checking for uncertainty columns.
        
        Args:
            dataset: Input dataset
            
        Returns:
            True if uncertainty columns are present
        """
        uncertainty_indicators = ['sigma_x', 'sigma_v', 'sigma_a', 'sigma_total']
        
        # Check if any uncertainty indicators are present
        if isinstance(dataset, dict):
            dataset_keys = set(dataset.keys())
            has_uncertainty = any(indicator in dataset_keys for indicator in uncertainty_indicators)
            
            if has_uncertainty:
                logging.info("Uncertainty task detected - switching to GPU mode")
                return True
        
        logging.info("Standard task detected - using CPU multiprocessing mode")
        return False
    
    def _should_use_gpu_mode(self, dataset: Dict[str, Any]) -> bool:
        """
        Determine whether to use GPU mode based on task characteristics.
        
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
    
    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
            inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> Tuple[Any, bool]:
        """
        Adaptive run method that chooses execution mode based on task requirements.
        
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
        dataset = inputs[test_input]
        
        # Determine execution mode
        use_gpu_mode = self._should_use_gpu_mode(dataset)
        
        if use_gpu_mode:
            logging.info("Using GPU sequential execution mode")
            return self._run_gpu_mode(program, function_to_run, function_to_evolve, 
                                    dataset, timeout_seconds, **kwargs)
        else:
            logging.info("Using CPU multiprocessing mode")
            return self._run_cpu_mode(program, function_to_run, function_to_evolve, 
                                    dataset, timeout_seconds, **kwargs)
    
    def _run_gpu_mode(self, program: str, function_to_run: str, function_to_evolve: str,
                     dataset: Dict[str, Any], timeout_seconds: int, **kwargs) -> Tuple[Any, bool]:
        """
        GPU mode: Sequential execution without multiprocessing.
        
        This mode is used for uncertainty-aware tasks that require GPU computation
        and automatic differentiation, which don't work well with multiprocessing.
        """
        try:
            # Set up timeout signal handler
            def timeout_handler(signum, frame):
                raise TimeoutError("GPU execution timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                # Execute directly in main process
                all_globals_namespace = {}
                exec(program, all_globals_namespace)
                function_to_run = all_globals_namespace[function_to_run]
                
                logging.info(f"GPU mode: Executing {function_to_run.__name__}")
                results = function_to_run(dataset)
                
                # Validate results
                if not isinstance(results, (tuple, list)) or len(results) != 2:
                    logging.error(f"Invalid evaluate output: expected tuple of (score, params), got {results}")
                    return (None, None, False)
                
                score, params = results
                if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                    logging.error(f"Invalid evaluate output types: score={type(score)}, params={type(params)}")
                    return (None, None, False)
                
                if isinstance(score, torch.Tensor):
                    score = score.item()
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                signal.alarm(0)  # Cancel timeout
                return (score, params, True)
                
            except TimeoutError:
                logging.error(f"GPU execution timed out after {timeout_seconds} seconds")
                return (None, None, False)
            
        except Exception as e:
            logging.error(f"GPU mode execution error: {e}")
            return (None, None, False)
        finally:
            signal.alarm(0)  # Ensure timeout is canceled
    
    def _run_cpu_mode(self, program: str, function_to_run: str, function_to_evolve: str,
                     dataset: Dict[str, Any], timeout_seconds: int, **kwargs) -> Tuple[Any, bool]:
        """
        CPU mode: Multiprocessing execution for parallel evaluation.
        
        This mode is used for standard tasks that don't require GPU computation
        and can benefit from multiprocessing parallelism.
        """
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, result_queue)
        )
        
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            process.terminate()
            process.join()
            logging.error(f"CPU process timed out after {timeout_seconds} seconds")
            return (None, None, False)
        else:
            results = self._get_results(result_queue)
            return results
    
    def _compile_and_run_function(self, program: str, function_to_run: str, 
                                 function_to_evolve: str, dataset: Dict[str, Any], 
                                 result_queue: multiprocessing.Queue):
        """
        Compile and run function in subprocess (CPU mode only).
        """
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            
            logging.info(f"CPU mode: Executing {function_to_run.__name__}")
            results = function_to_run(dataset)
            
            # Validate results
            if not isinstance(results, (tuple, list)) or len(results) != 2:
                logging.error(f"Invalid evaluate output: expected tuple of (score, params), got {results}")
                result_queue.put((None, None, False))
                return
            
            score, params = results
            if not isinstance(score, (int, float, torch.Tensor, np.floating)) or not isinstance(params, (list, tuple)):
                logging.error(f"Invalid evaluate output types: score={type(score)}, params={type(params)}")
                result_queue.put((None, None, False))
                return
            
            if isinstance(score, torch.Tensor):
                score = score.item()
            
            result_queue.put((score, params, True))
            
        except Exception as e:
            logging.error(f"CPU mode execution error: {e}")
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
    
    def set_force_cpu(self, force: bool):
        """
        Force CPU mode regardless of task characteristics.
        
        Args:
            force: If True, always use CPU mode
        """
        self._force_cpu = force
        logging.info(f"Force CPU mode: {force}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics and system information.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            'gpu_available': self._gpu_available,
            'force_cpu': self._force_cpu,
            'timeout_seconds': self._timeout_seconds,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'torch_version': torch.__version__,
        }

# Backward compatibility wrapper
class Evaluator(AdaptiveEvaluator):
    """
    Backward compatibility wrapper for the original Evaluator class.
    """
    
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        # Maintain original interface
        self._numba_accelerate = False
        
        # Original warning behavior
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, evaluate will run on CPU")
    
    def timeout_handler(self, signum, frame):
        """Original timeout handler for backward compatibility."""
        raise TimeoutError("Execution timed out")

# Factory function for easy instantiation
def create_evaluator(mode: str = 'adaptive', **kwargs) -> AdaptiveEvaluator:
    """
    Factory function to create evaluator instances.
    
    Args:
        mode: 'adaptive' for adaptive mode, 'cpu' for CPU-only, 'gpu' for GPU-preferred
        **kwargs: Additional arguments
        
    Returns:
        Evaluator instance
    """
    evaluator = AdaptiveEvaluator(**kwargs)
    
    if mode == 'cpu':
        evaluator.set_force_cpu(True)
    elif mode == 'gpu':
        evaluator.set_force_cpu(False)
    # 'adaptive' is the default
    
    return evaluator