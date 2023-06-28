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
conda env create --file=deepspeed.yaml
```

## Test Deepspeed env

```bash
conda activate ds
ds_report
```