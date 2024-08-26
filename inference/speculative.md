# Speculative Decoding

## MLP Speculator
```bash
  -e MODEL_ID=ibm-fms/llama3-8b-accelerator \


sudo docker run --gpus 2 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct \
  -e NUM_SHARD=2 \
  -e MAX_INPUT_TOKENS=1562 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:sha-b70ae09
```

send test request 

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": false,
  "max_tokens": 250
}' \
    -H 'Content-Type: application/json'
```

## Medusa Speculator

```bash
sudo docker run --gpus all -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=text-generation-inference/Mistral-7B-Instruct-v0.2-medusa \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=1562 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:sha-b70ae09
```

send test request 

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "user",
      "content": "Write a poem for my three year old"
    }
  ],
  "stream": false,
  "max_tokens": 250
}' \
    -H 'Content-Type: application/json'
```

# Benchmark

1. Run TGI container with the following command:
2. Run K6s benchmark with the following command:
  
  ```bash 
  k6 run -e VU=10 -e DATA=samples.json oai_load.js
  ```

# Results

`Acceleration = (Total Tokens (Count + Sum)) / (Number of Loops/Count)`

| Model                               | vu  | Throughput (req/s) | Token Sum | Tokens Count | Total Tokens | Acceleration |
| ----------------------------------- | --- | ------------------ | --------- | ------------ | ------------ | ------------ |
| ibm-fms/llama3-8b-accelerator       | 1   | 0.250124/s         | 783       | 1067         | 1850         | 1.73         |
| ibm-fms/llama3-8b-accelerator       | 10  | 0.934835/s         | 2953      | 3885         | 6838         | 1.76         |
| meta-llama/Meta-Llama-3-8B-Instruct | 1   | 0.23309/s          | 0         | 1400         | 1400         | 1.00         |
| meta-llama/Meta-Llama-3-8B-Instruct | 10  | 0.913191/s         | 0         | 6408         | 6408         | 1.00         |
