import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
import json
from dataclasses import dataclass, field
from pprint import pprint
from functools import partial

import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from multiprocessing import Pool
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any


@dataclass
class Arguments:
    input_file: str = field(
        default="./data/ultrafeedback_binarized_train_gen_reward_v4.jsonl"
    )
    output_file: str = field(
        default="./data/ultrafeedback_binarized_train_gen_reward_v4_tokenized.jsonl"
    )
    batch_size: int = field(default=8)
    model_name_or_path: str = field(default="/liziniu/models/RLHFlow-LLaMA3-SFT-v2")
    tokenizer_name_or_path: str = field(default="/liziniu/models/RLHFlow-LLaMA3-SFT-v2")
    max_seq_length: int = field(default=8192)
    preprocessing_num_workers: int = field(default=64)


@dataclass
class DataCollatorSFT:
    """
    Data collator for SFT that handles batching and padding of examples.
    """

    tokenizer: Any

    def __call__(self, features):
        batch = {}

        # Process input_ids
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        input_ids = self.zero_pad_sequences(input_ids)
        batch["input_ids"] = input_ids

        # Process attention_mask
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        attention_mask = self.zero_pad_sequences(attention_mask)
        batch["attention_mask"] = attention_mask

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


def init_worker(tokenizer_):
    global shared_tokenizer
    shared_tokenizer = tokenizer_


def encode_sft_example(
    example,
    max_seq_length,
    prompt_key="question",
    response_key="response",
    verbose=False,
    tokenizer=None,
):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    global shared_tokenizer
    # Use the passed tokenizer if available (for main process), otherwise use global tokenizer (for worker processes)
    tokenizer_to_use = tokenizer if tokenizer is not None else shared_tokenizer
    if tokenizer_to_use is None:
        raise ValueError("Tokenizer not found. Make sure it's properly initialized.")

    reward = example.get("reward", 1)
    if reward < 0:
        raise ValueError(
            f"It is not recommended to use negative reward: {reward}. Please consider to shift the reward to a positive range."
        )

    if "messages" in example:
        messages = example["messages"]
    else:
        messages = [
            {"role": "user", "content": example[prompt_key]},
            {
                "role": "assistant",
                "content": example[response_key],
            },
        ]

    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    if verbose:
        chat_messages = tokenizer_to_use.apply_chat_template(
            conversation=messages,
            tokenize=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )
        print(f"chat_messages:\n[{chat_messages}]")
        print(f"reward: {reward}")
    input_ids = tokenizer_to_use.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer_to_use.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer_to_use.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer_to_use.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "attention_mask": attention_mask.flatten().tolist(),
        "reward": reward,
    }


def main(args):
    pprint(args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    print(f"load tokenizer from {args.tokenizer_name_or_path} done.")
    max_seq_length = args.max_seq_length

    ############################
    # Compute log probabilities
    ############################

    input_data = []
    with open(args.input_file) as file:
        if args.input_file.endswith(".jsonl"):
            for line in file:
                example = json.loads(line)
                input_data.append(example)
        elif args.input_file.endswith(".json"):
            input_data = json.load(file)
        else:
            raise ValueError(f"Unsupported file extension: {args.input_file}")

    print(
        f"load input data from {args.input_file} done. len(input_data): {len(input_data)}"
    )

    ############################
    # Tokenize the dataset
    ############################

    # Tokenize the first example
    print(
        encode_sft_example(
            input_data[0], max_seq_length, verbose=True, tokenizer=tokenizer
        )
    )

    # Tokenize the entire dataset
    tokenized_data = []
    with Pool(
        args.preprocessing_num_workers, initializer=init_worker, initargs=(tokenizer,)
    ) as p:
        pbar = tqdm(input_data, desc=f"tokenizing")
        encode_fn = partial(encode_sft_example, max_seq_length=max_seq_length)
        for tokenized_example in p.imap(encode_fn, pbar):
            if tokenized_example is not None:
                dump = json.dumps(tokenized_example)
                tokenized_data.append(dump)

    ############################
    # Compute log probabilities
    ############################

    # Load model with device_map="auto" - this is the simplest approach for multi-GPU
    print("Loading model with device_map='auto' for multi-GPU processing...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",  # This will automatically distribute across available GPUs
    )

    print(
        f"Model loaded and distributed across GPUs with device map: {model.hf_device_map}"
    )
    model.eval()

    # Get the device of the first layer of the model to determine where to send inputs
    # This is a fallback approach if device_map="auto" doesn't handle it automatically
    first_device = next(model.parameters()).device
    print(f"First model parameter device: {first_device}")

    data_collator = DataCollatorSFT(tokenizer)
    with torch.no_grad():
        for i in tqdm(
            range(0, len(tokenized_data), args.batch_size),
            desc="Computing log probabilities",
        ):
            batch_data = [
                json.loads(x) for x in tokenized_data[i : i + args.batch_size]
            ]
            batch = data_collator(batch_data)

            # Move input tensors to the same device as the model's first layer
            # With device_map="auto", HuggingFace should handle the device placement internally
            input_ids = batch["input_ids"].to(first_device)
            attention_mask = batch["attention_mask"].to(first_device)

            # Process batch
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            log_probs = log_probs_from_logits(outputs.logits[:, :-1], input_ids[:, 1:])

            for j, log_prob in enumerate(log_probs):
                seq_length = len(batch_data[j]["input_ids"])
                tokenized_data[i + j] = batch_data[j]
                tokenized_data[i + j]["log_prob"] = (
                    log_prob[: seq_length - 1].cpu().tolist()
                )  # Move to CPU for serialization
                tokenized_data[i + j] = json.dumps(tokenized_data[i + j])

    ############################
    # Save the tokenized data
    ############################
    with open(args.output_file, "w") as fw:
        for dump in tokenized_data:
            fw.write(dump + "\n")
    print(f"Saved to {args.output_file}")


# Function to compute log probabilities from logits
def log_probs_from_logits(logits, labels):
    """
    Compute the log probabilities given the logits and labels.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)
