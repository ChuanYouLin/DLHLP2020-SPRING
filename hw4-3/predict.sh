python  ./examples/run_squad.py \
	--model_type bert \
	--do_eval \
	--model_name_or_path ./output/ROBERTA_MODEL \
	--tokenizer_name pre-train/hfl/chinese-roberta-wwm-ext \
	--train_file Data/hw4-3_train.json \
	--predict_file Data/hw4-3_dev.json \
	--max_seq_length 384 \
	--per_gpu_eval_batch_size 100 \
	--output_dir ./output/ROBERTA_MODEL \

# --tokenizer_name bert-base-chinese \
# --model_name_or_path ./output/ROBERTA_MODEL \
# --tokenizer_name pre-train/hfl/chinese-roberta-wwm-ext \
