# Offline REINFORCE

A PyTorch implementation of Offline REINFORCE for fine-tuning language models using reward-weighted likelihood estimation. This implementation is based on Section 4.1 of [ReMax Paper](https://arxiv.org/pdf/2310.10505) and serves as an efficient tool for rapid prototyping and validation of RL-based language model training approaches before scaling to full production training.

## Overview

This project implements Offline REINFORCE, a method for training language models using offline data with associated rewards. The approach is based on reward-weighted likelihood estimation with KL regularization to prevent the model from deviating too far from the reference model.

The implementation supports:
- Training with pre-tokenized datasets containing inputs, labels, and rewards
- KL regularization with a configurable coefficient
- Reward clipping to stabilize training
- Integration with HuggingFace Transformers for easy model loading and training

## Key Components

- **`trainer.py`**: Implements `OfflineREINFORCETrainer`, a custom trainer that extends the HuggingFace Trainer with reward-weighted likelihood estimation. The loss function combines rewards with a KL divergence term to regularize the policy.

- **`train.py`**: Contains the main training procedure, including argument parsing, dataset loading, model initialization, and training loop execution. It supports various training configurations via command-line arguments.

- **`process_dataset.py`**: Handles data preprocessing, converting raw examples into the tokenized format required for training. The output includes `input_ids`, `attention_mask`, `labels`, and `rewards` fields, along with reference log probabilities for KL regularization.

## Installation

```bash

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

Process your raw data into the required format using:

```bash
export CUDA_VISIBLE_DEVICES="0"
python process_dataset.py \
  --input_file path/to/input.jsonl \
  --output_file path/to/output.tokenized.jsonl \
  --model_name_or_path <model-name> \
  --tokenizer_name_or_path <tokenizer-name> \
  --max_seq_length 4096
```

### Training

Train a model using:

```bash
python train.py \
  --model_name_or_path <model-name> \
  --train_tokenized_file path/to/tokenized_data.jsonl \
  --output_dir path/to/save/model \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --kl_coeff 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 1
```

## Method

The Offline REINFORCE method implemented here:

1. Uses pre-computed rewards for each training example
2. Employs a KL regularization term to prevent the model from deviating too far from the reference model
3. Optimizes the policy to maximize the expected reward while staying close to the reference model

The loss function is defined as:
```
loss = -mean((rewards - kl_coeff * KL(policy, reference)) * log_probs)
```

where:
- `rewards` are clipped to a reasonable range for stability
- `KL` is the KL divergence between the current policy and the reference policy
- `kl_coeff` controls the strength of the KL regularization

## Credit

The multi-processing GPU part in `preprocess_dataset.py` is improved by Peter Chen from Columbia University.

