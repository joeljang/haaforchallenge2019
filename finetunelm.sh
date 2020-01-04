export TRAIN_FILE=./training.raw
export TEST_FILE=./testing.raw

python3 finetunelm.py \
    --output_dir=outputbert \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_lower_case \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --overwrite_output_dir \
    --learning_rate=1e-4 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm