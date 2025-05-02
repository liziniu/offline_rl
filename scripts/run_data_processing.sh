#!/bin/bash

set -e 
set -x 

export CUDA_VISIBLE_DEVICES="0,1"

python process_dataset.py \
    --input_file "./data/ultrafeedback_4k.jsonl" \
    --output_file "./data/ultrafeedback_4k_tokenized.jsonl" \
    --model_name_or_path "/liziniu/models/RLHFlow-LLaMA3-SFT-v2" \
    --tokenizer_name_or_path "/liziniu/models/RLHFlow-LLaMA3-SFT-v2" \
    --max_seq_length 8192 \
    --preprocessing_num_workers 64 \
    --batch_size 16
