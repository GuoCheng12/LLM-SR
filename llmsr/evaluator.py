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

from __future__ import annotations
import pdb
from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing
import torch
import numpy as np
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
    if not generated_code:
        return ''
    code = f'def fake_function_header():\n{generated_code}'
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])
    if not code:
        return ''
    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'

def _sample_to_program(generated_code: str, version_generated: int | None, template: code_manipulation.Program, function_to_evolve: str) -> tuple[code_manipulation.Function, str]:
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )
    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)

class Sandbox(ABC):
    @abstractmethod
    def run(self, program: str, function_to_run: str, function_to_evolve: str, inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        raise NotImplementedError('Must provide a sandbox for executing untrusted code.')

class LocalSandbox(Sandbox):
    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = False  # Disable Numba for GPU compatibility
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, evaluate will run on CPU")
    def timeout_handler(self, signum, frame):
        raise TimeoutError("Execution timed out")
    
    def run(self, program: str, function_to_run: str, function_to_evolve: str, inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        dataset = inputs[test_input]
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
            results = (None, None, False)
        else:
            results = self._get_results(result_queue)
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return (None, None, False)

    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Results: {results}\n=====================================================\n\n')

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate, result_queue):
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            logging.info(f"Executing {function_to_run.__name__} with dataset shape: {dataset['inputs'].shape}")
            results = function_to_run(dataset)
            
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
            logging.error(f"Execution Error in evaluate: {e}\nProgram:\n{program}\nDataset shapes: inputs={dataset['inputs'].shape}, outputs={dataset['outputs'].shape}")
            result_queue.put((None, None, False))

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False

class Evaluator:
    def __init__(self, database, template, function_to_evolve, function_to_run, inputs, timeout_seconds=60, sandbox_class=Sandbox):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(self, sample: str, island_id: int | None, version_generated: int | None, **kwargs) -> None:
        new_function, program = _sample_to_program(sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        params_per_test = {}
        time_reset = time.time()

        for current_input in self._inputs:
            score, params, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input, self._timeout_seconds
            )

            logging.info(f"Run result: score={score}, params={params}, runs_ok={runs_ok}")
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and score is not None:
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
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.params = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)