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

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib
import requests
import json
import os
import pdb
# Define models dictionary
models = {
    'GPT-4o': 'gpt-4o',
    'gpt-3.5-turbo': 'gpt-3.5-turbo'  # Added support for gpt-3.5-turbo
}

def get_send_request(MLLM='gpt-3.5-turbo'):
    if MLLM in models.keys():
        URL = "http://35.220.164.252:3888/v1/chat/completions"
        API_KEY = 'sk-n5EHLdSRE6upuHnoy6Ch20NAao7Z34gRp1aVXFOZLuuHAZ4p'  # Use environment variable or fallback
        HEADERS = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        def _send_request(messages):
            count = 0
            while count < 10:
                count += 1
                payload = json.dumps({
                    "model": models[MLLM],
                    "messages": messages,
                    "temperature": 0.75,
                    "max_tokens": 1024
                })
                session = requests.Session()
                session.keep_alive = False
                response = session.post(URL, headers=HEADERS, data=payload, verify=False)
                try:
                    content = response.json()['choices'][0]['message']['content']
                    return content
                except Exception as e:
                    print(f"Retry {count} times: {e}")
                    if count == 10:
                        print("Failed to get response after 10 retries.")
                        return ""
        return _send_request
    raise ValueError(f"Model {MLLM} not found in models dictionary")

class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1

def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        return code
    return sample

class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True, model='gpt-3.5-turbo') -> None:
        super().__init__(samples_per_prompt)
        self._url = "http://35.220.164.252:3888/v1/chat/completions"
        # self._instruction_prompt = (
        #     "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
        #     "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
        # )
        self._instruction_prompt = (
            "You are a helpful assistant in discovering the noise-robust mathematical functional structure of scientific systems. "
            "Complete the 'equation' function below, considering the physical meaning and relationships of inputs, and accounting for noisy inputs."
        )
        self._batch_inference = batch_inference
        self._trim = trim
        self._send_request = get_send_request(MLLM=model)  # Initialize with specified model

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt, config.api_model)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt, config.api_model)
                        all_samples.append(response)

                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception as e:
                print(f"Error in local inference: {e}")
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        final_prompt = '\n'.join([self._instruction_prompt, prompt])
        
        # # PDB断点: 查看发送给LLM的最终prompt
        # import pdb; pdb.set_trace()
        # print("=== DEBUG: Final prompt sent to LLM ===")
        # print(f"Final prompt length: {len(final_prompt)}")
        # print("First 1000 chars of final prompt:")
        # print(final_prompt[:1000])
        # print("=" * 60)
        # print("Last 1000 chars of final prompt:")
        # print(final_prompt[-1000:])
        # print("Contains @evaluate.run:", "@evaluate.run" in final_prompt)
        # print("Contains @equation.evolve:", "@equation.evolve" in final_prompt)
        # print("Contains 'def evaluate':", "def evaluate" in final_prompt)
        # print("=" * 60)
        # pdb.set_trace()
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    messages = [{"role": "user", "content": final_prompt}]
                    response = self._send_request(messages)
                    if not response:
                        raise Exception("Empty response from API")
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break
                except Exception as e:
                    print(f"Error during API call: {e}")
                    continue
        
        return all_samples
    
    def _do_request(self, content: str, model: str) -> str:
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        messages = [{"role": "user", "content": content}] * repeat_prompt
        response = self._send_request(messages)
        
        if response:
            if self._batch_inference:
                return [response] if isinstance(response, str) else response
            return response
        raise Exception("Failed to get valid response from local inference")