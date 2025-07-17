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

"""A multi-island experience buffer that implements the evolutionary algorithm."""
from __future__ import annotations
import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping, List
import pdb
from absl import logging
import numpy as np
import scipy.special
from llmsr import code_manipulation
from llmsr import config as config_lib


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]
ParamsPerTest = Mapping[Any, List[float]]

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result

def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)

def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))

@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int

class Island:
    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
            original_specification: str = None,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)
        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0
        self._original_specification = original_specification

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            params_per_test: ParamsPerTest | None = None,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)
        program.score = score
        program.params = params_per_test.get('data') if params_per_test else None
        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""

        signatures = list(self._clusters.keys())
        
        # 如果没有clusters，返回基础模板
        if not signatures:
            # 使用基础模板创建prompt
            base_function = self._template.get_function(self._function_to_evolve)
            prompt = self._generate_prompt([base_function])
            return prompt, 1
        
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        
        prompt = self._generate_prompt(sorted_implementations)
        
        return prompt, version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)

        # Extract evaluate.run function from original specification
        evaluate_text = None
        
        if self._original_specification:
            # Extract evaluate function with decorator from original text
            import re
            evaluate_match = re.search(r'@evaluate\.run\s*\ndef\s+evaluate\s*\([^)]*\)[^:]*:.*?(?=\n\n@|\n\ndef|\Z)', 
                                     self._original_specification, re.DOTALL)
            if evaluate_match:
                evaluate_text = evaluate_match.group(0)
                # Add debug info
                # print(f"DEBUG: Found evaluate function with decorator: {len(evaluate_text)} chars")
                # print(f"DEBUG: First 200 chars: {evaluate_text[:200]}")
        
        # Also extract equation function with decorator for comparison
        equation_text = None
        if self._original_specification:
            equation_match = re.search(r'@equation\.evolve\s*\ndef\s+equation\s*\([^)]*\)[^:]*:.*?(?=\n\n|\Z)', 
                                     self._original_specification, re.DOTALL)
            if equation_match:
                equation_text = equation_match.group(0)
                #print(f"DEBUG: Found equation function with decorator: {len(equation_text)} chars")
        
        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        
        # Add evaluate.run function FIRST if found
        if evaluate_text:
            # Create a custom function object that renders the original text
            class CustomFunction:
                def __init__(self, text):
                    self.text = text
                    self.name = "evaluate"
                def __str__(self):
                    return self.text
            
            evaluate_function = CustomFunction(evaluate_text)
            versioned_functions.append(evaluate_function)
            print(f"DEBUG: Added evaluate function to versioned_functions")
        else:
            print("DEBUG: No evaluate function found")
        
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create header of new function to be completed
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)
        
        # Replace functions in the template with the list constructed here.
        # Since we have mixed types (Function and CustomFunction), we need to handle them separately
        
        # Build the prompt manually
        prompt_parts = []
        
        # Add preface
        if self._template.preface:
            prompt_parts.append(self._template.preface)
        
        # Add all functions
        for func in versioned_functions:
            prompt_parts.append(str(func))
        
        final_prompt = '\n'.join(prompt_parts)
        
        # print(f"DEBUG: Final prompt contains {len(versioned_functions)} functions")
        # print(f"DEBUG: Function names: {[getattr(f, 'name', 'unknown') for f in versioned_functions]}")
        # print(f"DEBUG: Final prompt length: {len(final_prompt)}")
        # print(f"DEBUG: Contains @evaluate.run: {'@evaluate.run' in final_prompt}")
        # print(f"DEBUG: Contains def evaluate: {'def evaluate' in final_prompt}")
        
        return final_prompt

class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)

class ExperienceBuffer:
    def __init__(self, config: config_lib.ExperienceBufferConfig, template: code_manipulation.Program, function_to_evolve: str, original_specification: str = None) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._original_specification = original_specification
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period,
                       original_specification))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)
        self._best_params_per_island: list[List[float] | None] = (
                [None] * config.num_islands)
        self._last_reset_time = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, params_per_test: ParamsPerTest | None = None, **kwargs) -> None:
        self._islands[island_id].register_program(program, scores_per_test, params_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            self._best_params_per_island[island_id] = params_per_test.get('data') if params_per_test else None
        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.params = params_per_test.get('data') if params_per_test else None
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            params_per_test: ParamsPerTest | None = None,
            **kwargs 
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, params_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, params_per_test, **kwargs)
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
                self._original_specification)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            founder_params = self._best_params_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores, {'data': founder_params})