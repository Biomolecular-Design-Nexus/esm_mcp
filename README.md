# ESM MCP server

ESM MCP server for protein modeling, extracted from the official ESM tutorial.


## Overview

This MCP server provides five main tools for protein modeling:

1. **esm_extract_embeddings**:
2. **esm_train_fitness_predictor**:
3. **esm_predict_fitness_predictor**:
4. **esm_predict_likelihood**:
5. **esmif_predict_likelihood**:

## Installation

```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.10 pip -y
mamba activate ./env

# Install dependencies
pip install -r requirements.txt
pip install --ignore-installed fastmcp
```

## Local usage

### 1. Extracting ESM embeddings
```shell
# 5-fold cross validation
python repo/ev_onehot/train.py example/ --cross_val

# Train test split training
python repo/ev_onehot/train.py example/  -s 1
```

### 2. Training fitness Models
```shell
# 5-fold cross validation
python repo/ev_onehot/train.py example/ --cross_val

# Train test split training
python repo/ev_onehot/train.py example/  -s 1
```

### 3. Making Predictions

```shell
python repo/ev_onehot/pred.py example --seq_path example/data.csv
```


## MCP usage
### Install `ev+onehot` mcp
```shell
fastmcp install claude-code mcp-servers/ev_onehot_mcp/src/ev_onehot_mcp.py --python mcp-servers/ev_onehot_mcp/env/bin/python
```

## Call MCP