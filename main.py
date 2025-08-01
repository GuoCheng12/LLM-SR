
import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd
import pdb
from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)

args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,)
    global_max_sample_num = 10000 

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    df = pd.read_csv('./data/' + problem_name + '/train.csv')
    data = np.array(df)
    
    if "noise" in problem_name:
        # For noise problems, X = [t, x, v], y = [a, sigma_a]
        X = data[:, [0,1,2,4,5]]  # t, x, v, sigma_x, sigma_v
        y = data[:, [3, 6]]  # a and sigma_a
    else:
        # For non-noise problems, X = [t, x, v], y = a
        X = data[:, :3]  # t, x, v
        y = data[:, -1]  # a
    
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)

    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=args.log_path,
    )