# FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion

## Set up environment

Run the following commands to create a conda environment:
```bash
conda create -n MulEHR python=3.8
source activate MulEHR
pip install -r requirements.txt
```
## Run the model

MIMIC-III experiments:
```
sh run.sh
```

MIMIC-III experiments:
```
sh run_mimiciv.sh
```

## Load the results
First change the `filepath` in `load_result.py`, then run
```
python load_result.py
```