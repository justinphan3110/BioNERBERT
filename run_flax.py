import os
os.environ['MODEL']='bert-base-cased'
os.environ['MODEL_PATH']=f"{os.environ['MODEL']}"

# os.environ['MODEL']='bert-base-cased'
# os.environ['MODEL_PATH']=f"out/pretraining/{os.environ['MODEL']}"

os.environ['task']='pretraining'
os.environ['datadir']=f"data"
os.environ['outdir']=f"out/{os.environ['task']}/{os.environ['MODEL']}"
os.environ["WANDB_DISABLED"] = "true"

os.system(f'''python3 pretrain/run_mlm_flax.py --model_name_or_path {os.environ['MODEL_PATH']} \
    --train_file {os.environ['datadir']}/pubmed_abs_biolinkner_synthetic_data.json --do_train --do_eval \
    --per_device_train_batch_size 128 --learning_rate 6e-4 \
    --warmup_steps 100 --max_seq_length 512 --num_train_epochs 1 --preprocessing_num_workers 16 \
    --save_strategy no --evaluation_strategy steps --eval_steps 500 --logging_steps 500 \
     --output_dir {os.environ['outdir']} --cache_dir /mnt/disks/persist --overwrite_output_dir''')