# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM-SR is a scientific equation discovery and symbolic regression framework that combines Large Language Models (LLMs) with evolutionary search to discover accurate and interpretable equations from data. The system uses LLMs to generate equation program skeletons, which are then evaluated against scientific datasets to evolve better hypotheses.

## Key Components

### Core Architecture
- **Pipeline** (`llmsr/pipeline.py`): Main orchestration system that coordinates samplers, evaluators, and the experience buffer
- **Sampler** (`llmsr/sampler.py`): Manages LLM interactions to generate equation hypotheses using both local and API-based models
- **Evaluator** (`llmsr/evaluator.py`): Executes generated code in sandboxed environments and scores equation hypotheses against datasets
- **Experience Buffer** (`llmsr/buffer.py`): Multi-island evolutionary algorithm that maintains populations of equation hypotheses and generates prompts for the LLM
- **Configuration** (`llmsr/config.py`): Centralized configuration management for experiment parameters

### LLM Integration
The system supports both local and API-based LLM inference:
- **Local LLMs**: Served via `llm_engine/engine.py` using HuggingFace models
- **API LLMs**: OpenAI GPT models via API calls
- **Prompt Engineering**: Equation skeletons are presented as programming problems with scientific context

### Evolutionary Search
- **Islands**: Multiple populations evolve independently with periodic resets
- **Clusters**: Programs with similar performance signatures are grouped together
- **Selection**: Programs are sampled based on performance scores and code length preferences

## Common Commands

### Environment Setup
```bash
# Create conda environment
conda create -n llmsr python=3.11.7
conda activate llmsr

# Install dependencies (choose one)
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate llmsr
```

### Running Local LLM Server
```bash
# Start local LLM server
cd llm_engine
python engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 --gpu_ids 0 --port 5000 --quantization

# Or use the provided script
bash run_server.sh
```

### Running LLM-SR Experiments

#### API-based (OpenAI GPT)
```bash
export API_KEY=[YOUR_API_KEY_HERE]

python main.py --use_api True \
               --api_model "gpt-3.5-turbo" \
               --problem_name oscillator1 \
               --spec_path ./specs/specification_oscillator1_numpy.txt \
               --log_path ./logs/oscillator1_gpt3.5
```

#### Local LLM
```bash
python main.py --problem_name oscillator1 \
               --spec_path ./specs/specification_oscillator1_numpy.txt \
               --log_path ./logs/oscillator1_local
```

### Available Problems and Datasets
- `oscillator1`: Simple harmonic oscillator
- `oscillator2`: Damped harmonic oscillator
- `oscillator_noise`: Noisy oscillator data
- `bactgrow`: Bacterial growth dynamics
- `stressstrain`: Stress-strain relationships

## Specification Files

Located in `specs/` directory, these files define the problem templates:
- Contain `@evaluate.run` decorated functions for fitness evaluation
- Contain `@equation.evolve` decorated functions as evolution targets
- Support both numpy (with scipy BFGS) and torch (with Adam) optimizers
- Include scientific context and parameter descriptions

## Data Format

Training data in `data/` follows CSV format:
- Standard problems: `[t, x, v, a]` (time, position, velocity, acceleration)
- Noise problems: `[t, x, v, a, sigma_x, sigma_v, sigma_a]` (with uncertainty estimates)

## Configuration Parameters

Key parameters in `llmsr/config.py`:
- `samples_per_prompt`: Number of hypotheses generated per LLM call
- `num_islands`: Population diversity through multiple evolutionary islands
- `functions_per_prompt`: Number of previous hypotheses included in prompts
- `evaluate_timeout_seconds`: Timeout for hypothesis evaluation
- `cluster_sampling_temperature_init`: Controls exploration vs exploitation

## Logging and Monitoring

- TensorBoard logs stored in specified `--log_path`
- Sample JSON files for detailed hypothesis tracking
- Profiling data for performance analysis
- Error logs in `evaluator.log` and `buffer.log`

## Development Notes

### Code Structure
- Uses `@evaluate.run` and `@equation.evolve` decorators to identify key functions
- Employs AST manipulation for code generation and modification
- Supports both CPU and GPU execution with CUDA optimization
- Implements multiprocessing for safe code execution with timeouts

### LLM Prompting Strategy
- Includes scientific context and physical meaning guidance
- Provides examples of successful equation structures
- Uses evolutionary feedback to improve subsequent generations
- Balances exploration of new structures with exploitation of promising candidates

### Testing and Validation
- No formal test suite present - validation occurs through scientific benchmarks
- Hypothesis evaluation includes both accuracy and parameter fitting
- Out-of-domain generalization testing on held-out datasets
- Performance comparison against traditional symbolic regression methods

## Port Configuration

When using local LLM server, ensure port consistency:
- Default server port: 5000 (configurable in `run_server.sh`)
- Update URL in `sampler.py` if using different port
- API calls use external OpenAI endpoints (no local port needed)