#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
from typing import Optional
from functools import partial
import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, List, Union

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainingArguments,
)

from trainer import OfflineREINFORCETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


@dataclass
class DataArguments:
    train_tokenized_file: str = field(
        default="./data/toy_data.tokenized.jsonl",
        metadata={"help": "huggingface dataset name or local data path"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    kl_coeff: float = field(
        default=0.0, metadata={"help": "KL regularization coeffient"}
    )


class CustomDataset(Dataset):
    def __init__(
        self,
        training_args,
        data_args,
        model_args,
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

        raw_datasets = load_dataset(
            "json",
            data_files=[self.data_args.train_tokenized_file],
            cache_dir=self.model_args.cache_dir,
        )
        self.data = raw_datasets["train"]

        if self.data_args.max_train_samples is not None:
            max_samples = min(len(self.data), self.data_args.max_train_samples)
            self.data = self.data.select(range(max_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        assert "input_ids" in example
        assert "labels" in example
        assert "reward" in example
        # example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
        example = {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
            "reward": torch.tensor(example["reward"], dtype=torch.float32),
            "ref_log_prob": torch.tensor(example["log_prob"], dtype=torch.float32),
        }
        return example


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        rewards = [x.pop("reward") for x in features]
        ref_log_probs = [x.pop("ref_log_prob") for x in features]
        batch = super().__call__(features, return_tensors)
        batch["reward"] = torch.stack(rewards)

        ref_log_probs = self.zero_pad_sequences(ref_log_probs, side="right", value=0.0)
        batch["ref_log_prob"] = self.pad_to_length(
            ref_log_probs, length=batch["input_ids"].size(-1) - 1, pad_value=0.0
        )
        return batch

    @staticmethod
    def zero_pad_sequences(sequences, side: str = "right", value=0):
        assert side in ("left", "right")
        max_len = max(seq.size(-1) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - seq.size(-1)
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            padded_sequences.append(F.pad(seq, padding, value=value))
        return torch.stack(padded_sequences, dim=0)

    @staticmethod
    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [
                    tensor,
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    if local_rank == 0:

        def remove_non_serializable(data):
            serializable_data = {}
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                    serializable_data[key] = value
                except (TypeError, ValueError):
                    pass
            return serializable_data

        args_dict = remove_non_serializable(training_args.__dict__)
        args_dict.update(remove_non_serializable(data_args.__dict__))
        args_dict.update(remove_non_serializable(model_args.__dict__))

        json.dump(
            dict(sorted(args_dict.items(), key=lambda x: x[0])),
            open(os.path.join(training_args.output_dir, "args.json"), "w"),
            indent=2,
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    global_rank = dist.get_rank()
    logger.warning(
        f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = len(tokenizer) - 1
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=(
            "flash_attention_2" if model_args.use_flash_attn else "eager"
        ),
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # set up datasets
    train_dataset = CustomDataset(training_args, data_args, model_args)

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    trainer = OfflineREINFORCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
