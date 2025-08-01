# profile the experiment with tensorboard
from heapq import heappush, heapreplace


import os.path
from typing import List, Dict
import logging
import json
from llmsr import code_manipulation
from torch.utils.tensorboard import SummaryWriter
import pdb
from typing import List, Dict, Tuple

class Profiler:
    def __init__(self, log_dir: str | None = 'runs', pkl_dir: str | None = None, max_log_nums: int | None = 10000):
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        self._top_3_scores: List[Dict[str, any]] = []

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

    def _write_tensorboard(self):
        if not self._log_dir or self._cur_best_program_str is None:
            return
        self._writer.add_scalar('Best Score of Function', self._cur_best_program_score, global_step=self._num_samples)
        self._writer.add_scalars('Legal/Illegal Function', {
            'legal function num': self._evaluate_success_program_num,
            'illegal function num': self._evaluate_failed_program_num
        }, global_step=self._num_samples)
        self._writer.add_scalars('Total Sample/Evaluate Time', {
            'sample time': self._tot_sample_time,
            'evaluate time': self._tot_evaluate_time
        }, global_step=self._num_samples)
        # Extract equation skeleton for the best program
        if self._cur_best_program_sample_order is not None and self._cur_best_program_sample_order in self._all_sampled_functions:
            best_program = self._all_sampled_functions[self._cur_best_program_sample_order]
            best_equation_skeleton = self._extract_equation_body(best_program)
            self._writer.add_text('Best Function String', self._cur_best_program_str, global_step=self._num_samples)
            self._writer.add_text('Best Equation Skeleton', best_equation_skeleton, global_step=self._num_samples)
        else:
            self._writer.add_text('Best Function String', self._cur_best_program_str, global_step=self._num_samples)
        if self._top_3_scores:
            # 格式化top_3_scores显示，包含equation信息
            top_3_text = "\n".join([
                f"Rank {i+1}: Score={entry['score']:.6f}, Sample={entry['sample_order']}\n"
                f"Equation: {entry['function']}\n" 
                for i, entry in enumerate(self._top_3_scores)
            ])
            self._writer.add_text('Top_3_Scores', top_3_text, global_step=self._num_samples)

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums if programs.global_sample_nums is not None else 0
        function_str = str(programs)
        score = programs.score
        params = programs.params
        content = {
            'sample_order': sample_order,
            'function': self._extract_equation_body(programs),
            'score': score,
            'params': params,
            'top_3_scores': self._top_3_scores
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w', encoding='utf-8') as json_file:
            json.dump(content, json_file, indent=4, ensure_ascii=False)

    def _extract_equation_body(self, programs: code_manipulation.Function) -> str:
        """
        Extract the complete mathematical equation skeleton from the function body.
        """
        try:
            function_str = str(programs)
            lines = function_str.split('\n')
            
            # Extract the complete function body excluding header
            equation_skeleton = []
            inside_function = False
            indent_level = None
            
            for line in lines:
                stripped = line.strip()
                
                # Skip function definition line
                if stripped.startswith('def ') and 'equation(' in stripped:
                    inside_function = True
                    continue
                    
                if inside_function and stripped:
                    # Determine the base indentation level from the first non-empty line
                    if indent_level is None and line.lstrip() != line:
                        indent_level = len(line) - len(line.lstrip())
                    
                    # Skip docstrings and comments
                    if (stripped.startswith('"""') or stripped.startswith("'''") or 
                        stripped.startswith('#') or stripped.startswith('Args:') or 
                        stripped.startswith('Returns:') or stripped.startswith('Return:')):
                        continue
                    
                    # Include meaningful code lines
                    if (not stripped.startswith('import ') and 
                        not stripped.startswith('from ') and
                        stripped not in ['', '"""', "'''"]):
                        
                        # Remove the base indentation to make it cleaner
                        if indent_level and line.startswith(' ' * indent_level):
                            clean_line = line[indent_level:]
                        else:
                            clean_line = line.lstrip()
                            
                        equation_skeleton.append(clean_line)
            
            if equation_skeleton:
                # Join the skeleton lines
                skeleton_str = '\n'.join(equation_skeleton)
                
                # If it's just a simple return statement, extract the expression
                if len(equation_skeleton) == 1 and equation_skeleton[0].startswith('return '):
                    return equation_skeleton[0].replace('return ', '').strip()
                else:
                    # Return the complete skeleton for complex functions
                    return skeleton_str
            else:
                return f"equation_v{programs.global_sample_nums if programs.global_sample_nums else 0}"
                
        except Exception as e:
            logging.warning(f"Failed to extract equation body: {e}")
            return f"equation_v{programs.global_sample_nums if programs.global_sample_nums else 0}"
    
    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return
        sample_orders: int = programs.global_sample_nums if programs.global_sample_nums is not None else 0
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            if programs.score is not None:
                # 添加到top_3_scores，包含完整的equation信息
                score_entry = {
                    'score': programs.score,
                    'sample_order': sample_orders,
                    'function': self._extract_equation_body(programs),
                    'params': programs.params
                }
                self._top_3_scores.append(score_entry)
                # 按score降序排列，保留前3名
                self._top_3_scores.sort(key=lambda x: x['score'], reverse=True)
                self._top_3_scores = self._top_3_scores[:3]
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        equation_skeleton = self._extract_equation_body(function)
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        print(f'================= Evaluated Function =================')
        print(f'Equation Skeleton:')
        print(f'{equation_skeleton}')
        print(f'------------------------------------------------------')
        print(f'Full Function:')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Params       : {str(function.params)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        # 格式化显示top_3_scores
        if self._top_3_scores:
            print(f'Top 3 scores :')
            for i, entry in enumerate(self._top_3_scores):
                print(f'  Rank {i+1}: Score={entry["score"]:.6f}, Sample={entry["sample_order"]}')
                print(f'    Equation: {entry["function"]}')
        else:
            print(f'Top 3 scores : []')
        print(f'======================================================\n\n')
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1
        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time