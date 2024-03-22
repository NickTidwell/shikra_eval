MODEL_PATH=/home/ntidwell/git/shikra/weights
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        config/shikra_eval_multi_rec.py \
        --cfg-options model_args.model_name_or_path=$MODEL_PATH