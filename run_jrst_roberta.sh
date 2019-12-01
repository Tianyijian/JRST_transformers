export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROBERTA_PATH="/users5/yjtian/tyj/SA/RoBERTa/RoBERTa_zh_Large_PyTorch"
export DATA_DIR="/users5/yjtian/tyj/SA/JRST_transformers/data3"
export OUT_DIR="/users5/yjtian/tyj/SA/JRST2/roberta_jrst_r2_th_0"
export TASK_NAME=JRST2

python run_jrst.py \
    --model_type bert \
    --model_name_or_path $ROBERTA_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --data_dir $DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=1   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUT_DIR \
    --evaluate_during_training \
    --save_steps 1000 \
    --logging_steps 50 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 16 \
    --seed 100
