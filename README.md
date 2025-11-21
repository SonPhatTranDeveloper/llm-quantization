# LLM Quantization

A project for testing model quantization with Large Language Models (LLMs) using HuggingFace Transformers and BitsAndBytes.

## Overview

This project provides a simple interface for loading quantized LLM models, fine-tuning them with LoRA/QLoRA, and generating text. It supports 4-bit and 8-bit quantization using BitsAndBytes, which significantly reduces memory requirements while maintaining reasonable model performance. The training pipeline uses parameter-efficient LoRA/QLoRA fine-tuning for efficient model adaptation.

## Prerequisites

- Python >= 3.9
- CUDA-capable GPU (recommended for optimal performance)
- `uv` package manager (or pip)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-quantization
```

2. Install dependencies using `uv`:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Install additional dependencies for quantization:
```bash
pip install bitsandbytes accelerate hf_transfer
```

## Project Structure

```
llm-quantization/
├── config/
│   ├── quantization.yaml       # Quantization settings
│   ├── text_generation.yaml    # Text generation parameters
│   └── training.yaml           # Training configuration
├── data/
│   └── kinhvanhoa.txt          # Training data
├── src/
│   ├── models/
│   │   ├── load.py             # Model loading utilities
│   │   ├── text_generation.py  # Text generation class
│   │   └── trainer.py          # Training utilities
│   ├── utils/
│   │   ├── data.py             # Data preprocessing utilities
│   │   └── logging.py          # Logging utilities
│   └── scripts/
│       ├── text_generation.py  # Main script for text generation
│       └── train.py             # Main script for training
└── main.py                     # Entry point
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

### Model Configuration (`config/models/qwen2.5-3B.yaml`)
- `model_name`: HuggingFace model identifier (default: "Qwen/Qwen2.5-3B")

### Quantization Configuration (`config/quantization.yaml`)
- `quantization`: Quantization bits (4 or 8)

### Text Generation Configuration (`config/text_generation.yaml`)
- `prompt`: Input text prompt
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (0.0-1.0)
- `top_p`: Nucleus sampling threshold
- `top_k`: Top-k sampling parameter

### Training Configuration (`config/training.yaml`)
- `data`: Path to training data file
- `block_size`: Maximum sequence length for tokenized chunks
- `train_split`: Fraction of data to use for training (rest for validation)
- `training`: Training hyperparameters (learning rate, batch size, epochs, etc.)
- `lora`: LoRA/QLoRA configuration (rank, alpha, target modules, dropout)

## Running the Project

### Text Generation

Run text generation with default settings:
```bash
python src/scripts/text_generation.py
```

### Training

Run training with default settings:
```bash
python src/scripts/train.py
```

### Customizing Configuration

You can override configuration parameters via command line:

**Change the prompt:**
```bash
python src/scripts/text_generation.py prompt="What is machine learning?"
```

**Change quantization bits:**
```bash
python src/scripts/text_generation.py quantization=8
```

**Change generation parameters:**
```bash
python src/scripts/text_generation.py max_new_tokens=512 temperature=0.9 top_k=40
```

**Combine multiple parameters:**
```bash
python src/scripts/text_generation.py prompt="Explain quantum computing" quantization=8 max_new_tokens=256 temperature=0.8
```

### Using Different Models

To use a different model, override the model name:

```bash
python src/scripts/text_generation.py model_name="meta-llama/Llama-2-7b-hf"
```

```bash
python src/scripts/train.py model_name="meta-llama/Llama-2-7b-hf"
```

### Customizing Training

You can override training parameters via command line:

**Change learning rate and batch size:**
```bash
python src/scripts/train.py training.learning_rate=1e-4 training.per_device_train_batch_size=8
```

**Change LoRA parameters:**
```bash
python src/scripts/train.py lora.r=32 lora.lora_alpha=64
```

**Change data path:**
```bash
python src/scripts/train.py data=data/my_data.txt
```

## Configuration Examples

### Example 1: 4-bit Quantization with Custom Prompt
```bash
python src/scripts/text_generation.py \
    quantization=4 \
    prompt="Write a short story about a robot" \
    max_new_tokens=200 \
    temperature=0.8
```

### Example 2: 8-bit Quantization with Higher Temperature
```bash
python src/scripts/text_generation.py \
    quantization=8 \
    prompt="Explain the theory of relativity" \
    temperature=0.9 \
    top_p=0.95
```

### Example 3: Training with QLoRA
```bash
python src/scripts/train.py \
    quantization=4 \
    training.learning_rate=2e-4 \
    training.num_epochs=5 \
    lora.r=16 \
    lora.lora_alpha=32
```

### Example 4: Training with Custom Data
```bash
python src/scripts/train.py \
    data=data/my_custom_data.txt \
    training.num_epochs=3 \
    training.per_device_train_batch_size=8
```

## Development

Install development dependencies:
```bash
uv sync --group dev
```

Run linting:
```bash
ruff check .
```

## Dependencies

- `transformers`: HuggingFace Transformers library
- `hydra-core`: Configuration management
- `omegaconf`: Configuration object handling
- `datasets`: Dataset handling and preprocessing
- `peft`: Parameter-Efficient Fine-Tuning (LoRA/QLoRA)
- `accelerate`: Model acceleration utilities
- `bitsandbytes`: Quantization library

