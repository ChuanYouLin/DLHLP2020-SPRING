#! /bin/bash

OUTPUT_MODEL_PATH=$1
TEST_DATA_PATH=$2
OUTPUT_FILE_PATH=$3

CUDA_VISIBLE_DEVICES=0 python ./examples/run_squad.py \
        --model_type bert \
        --do_eval \
        --model_name_or_path $OUTPUT_MODEL_PATH \
        --tokenizer_name pre-train/hfl/chinese-roberta-wwm-ext \
        --config_name config.json \
        --train_file $TEST_DATA_PATH \
        --predict_file $TEST_DATA_PATH \
        --max_seq_length 384 \
        --per_gpu_eval_batch_size 100 \
        --output_dir ./ \

python process_ans.py ./predictions_.json $OUTPUT_FILE_PATH