# Mixtral 8x7B GPTQ

```bash
model=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
num_shard=1
max_input_length=3500
max_total_tokens=4096
quantize=gptq

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e EXLLAMA_VERSION=1 \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -e QUANTIZE=$quantize \
   ghcr.io/huggingface/text-generation-inference:1.3.1
```

send test request 

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] hat is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```