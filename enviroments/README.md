# Micromamba setup 

## Installation [REF](https://mamba.readthedocs.io/en/latest/installation.html)

1. get bin

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

2. Add to path and Update `.bashrc`
```
./bin/micromamba shell init -s bash -p ~/micromamba  # this writes to your .bashrc file
source ~/.bashrc
```

3. Add Alias
```
echo  'alias conda=micromamba' >> ~/.bashrc 
source ~/.bashrc
```

## Create enviornment

```bash
conda env create --file=/fsx/philipp/deep-learning-pytorch-huggingface/enviroments/deepspeed.yaml
```

## Test Deepspeed env

```bash
conda activate ds
ds_report
```

## Flash attention setup

```bash
conda env create --file=/fsx/philipp/deep-learning-pytorch-huggingface/enviroments/flash-deepspeed.yaml
```

## Test Flash attention env

```bash
conda env create --file=flash-deepspeed.yaml
conda activate ds-flash
pip install flash-attn --no-build-isolation
ds_report
```

## Manual Flash attention env 

```bash
conda create --name flash python=3.10 -c conda-forge
conda install cudatoolkit=11.7 -c nvidia -c conda-forge
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
conda install gxx_linux-64==11.* -c conda-forge
conda activate flash
pip install ninja packaging
pip install flash-attn --no-build-isolation
pip install transformers[sklearn,sentencepiece]==4.31.0 accelerate==0.21.0 datasets==2.14.0 deepspeed==0.9.5 peft==0.4.0 evaluate einops loralib
``````