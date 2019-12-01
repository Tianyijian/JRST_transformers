export CUDA_VISIBLE_DEVICES=0,1,2,3
export BERT_PATH="/users5/yjtian/tyj/SA/BERT-torch/bert-base-chinese"
export MODEL_PATH="/users5/yjtian/tyj/SA/JRST2/bert_jrst_r2_th_0/fold-0-checkpoint-1125"
export DATA_DIR="/users5/yjtian/tyj/SA/JRST_transformers/data2"
export OUT_DIR="/users5/yjtian/tyj/SA/JRST2/bert_jrst_r2_th_0"
export TASK_NAME=JRST2

python run_jrst.py \
    --model_type bert \
    --model_name_or_path $MODEL_PATH \
    --task_name $TASK_NAME \
    --do_test \
    --data_dir $DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir $OUT_DIR \
    --evaluate_during_training \
    --save_steps 200 \
    --logging_steps 50 \
    --max_steps -1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.001 \
    --gradient_accumulation_steps 2 \
    --seed 10
