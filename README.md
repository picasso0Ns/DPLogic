# DPLogic

# Differentiable Probabilistic Logic Reasoning for Knowledge Graph Completion

This repository contains an implementation of the approach described in the paper, using relation-rule joint embedding to enhance knowledge graph completion with probabilistic logic reasoning.

## Overview

The approach consists of three main components:
1. **Predicate Logic Representation**: Efficiently constructs MLN structure for KG completion by constraining grounding to 2-hop enclosing subgraphs.
2. **Relation-Rule Joint Embedding**: Represents rule weights through relation embeddings using a neural logical operator, enabling differentiable optimization.
3. **Joint Optimization Framework**: Iteratively updates the joint distribution within the EM algorithm, facilitating optimization of both embedding-based and probabilistic logic-based distributions.

## Setup

### Requirements

- Python 3.7+
- PyTorch 1.9+
- NetworkX
- NumPy
- tqdm
- matplotlib

You can install the requirements using pip:

```bash
pip install torch numpy networkx tqdm matplotlib
```

### Project Structure

The codebase is organized as follows:

```
.
├── main.py               # Main entry point
├── utils.py              # Utility functions
├── models/               # Model implementations
│   ├── transH.py         # TransH embedding model
│   ├── predicate_logic.py # Predicate logic representation
│   └── relation_rule_embedding.py # Relation-rule joint embedding
├── data_utils/           # Data utilities
│   └── dataset.py        # Dataset loader
├── optimization/         # Optimization components
│   └── joint_optimization.py # Joint EM-based optimization
├── checkpoints/          # Model checkpoints
└── results/              # Evaluation results
```

## Usage

### Data Format

The datasets should be organized in the following directory structure:

```
dataset_name/
├── entities.txt     # (Optional) One entity per line
├── relations.txt    # One relation per line
├── train.txt        # Training triples (h \t r \t t)
├── test.txt         # Testing triples (h \t r \t t)
└── facts.txt        # (Optional) All facts for rule mining
```

### Training and Evaluation

To train and evaluate the model on a dataset, run:

```bash
python main.py --dataset FB15k237 --epochs 1000 --batch_size 128 --lr 0.001
```

### Main Parameters

- `--dataset`: Dataset name (`family`, `kinship`, `UMLS`, `FB15k237`, `WN18RR`)
- `--dim`: Embedding dimension (default: 100)
- `--hidden_dim`: Hidden dimension for logical operator (default: 200)
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--margin`: Margin for TransH loss (default: 1.0)
- `--neg_ratio`: Negative sampling ratio (default: 10)
- `--rule_threshold`: Confidence threshold for rule selection (default: 0.5)
- `--max_rule_length`: Maximum length of rules to mine (default: 2)
- `--alpha`: Weight for representation loss (default: 1.0)
- `--beta`: Weight for distribution loss (default: 0.5)
- `--gamma`: Weight for weight loss (default: 0.3)
- `--seed`: Random seed (default: 42)
- `--gpu`: GPU ID, -1 for CPU (default: 0)
- `--save_model`: Save model after training

## Datasets

The implementation supports the following datasets:
- `family`: Family relationships
- `kinship`: Kinship relationships
- `UMLS`: Unified Medical Language System
- `FB15k237`: A subset of Freebase
- `WN18RR`: A subset of WordNet

## Example
