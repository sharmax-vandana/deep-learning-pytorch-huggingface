requirements.txt

```bash
transformers==4.30.2
datasets==2.12.0
accelerate==0.20.3
torch==2.0.1
bitsandbytes==0.39.0
git+https://github.com/huggingface/peft.git@189a6b8e357ecda05ccde13999e4c35759596a67
tensorboard 
einops 
loralib
```



Single GPU LoRA

```bash
python training/scripts/run_clm_fsdp_lora.py \
  --model_id tiiuae/falcon-7b \
  --trust_remote_code True \
  --dataset_path "data" \
  --target_modules "query_key_value" "dense" "dense_h_to_4h" "dense_4h_to_h" \
  --output_dir ./tmp \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --optim adamw_torch_fused \
  --learning_rate 2e-4 \
  --gradient_checkpointing True \
  --bf16 True \
  --tf32 True \
  --logging_steps 10
```


Single GPU, LoRA + Int4

```bash
python training/scripts/run_clm_fsdp_lora.py \
  --model_id tiiuae/falcon-7b \
  --trust_remote_code True \
  --dataset_path "data" \
  --use_4bit True \
  --target_modules "query_key_value" "dense" "dense_h_to_4h" "dense_4h_to_h" \
  --output_dir ./tmp \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --optim adamw_torch_fused \
  --learning_rate 2e-4 \
  --gradient_checkpointing True \
  --bf16 True \
  --tf32 True \
  --logging_steps 10
```


Multi-GPU FSDP only -> not working

```bash
torchrun --nproc_per_node 8 run_clm.py \
  --dataset_path /opt/ml/input/data/training \
  --model_id togethercomputer/GPT-NeoXT-Chat-Base-20B \
  --output_dir /opt/ml/model \
  --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap GPTNeoXLayer \
  --target_modules "query_key_value" \
  --epochs 3 \
  --per_device_train_batch_size 2 \
  --bf16 True 
```

Multi-GPU Deepspeed

```bash
torchrun --nproc_per_node 8 run_clm.py \
  --dataset_path /opt/ml/input/data/training \
  --model_id togethercomputer/GPT-NeoXT-Chat-Base-20B \
  --output_dir /opt/ml/model \
  --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap GPTNeoXLayer \
  --epochs 3 \
  --per_device_train_batch_size 2 \
  --bf16 True 
```