# Federated Retrieval Augmentation over Heterogeneous Knowledge Graphs

This repository contains the implementation and experiments for the paper
"Federated Retrieval Augmentation over Heterogeneous Knowledge Graphs".

## Repository Status

**Released.**  The full training, alignment and evaluation code is available
in this repository.

## Overview

We study retrieval augmentation when the participating parties hold
knowledge graphs with different schemas and coverage.  The release includes a
privacy-preserving entity-alignment protocol, a federated dense retriever and
a cross-graph re-ranker.

## Repository Structure

```
├── src/
│   ├── model.py          # Federated retriever + re-ranker
│   ├── retriever.py      # Dense passage retrieval over local graphs
│   ├── federated.py      # Privacy-preserving alignment protocol
│   ├── train.py          # Training entry point
│   └── utils.py          # Data and metric helpers
├── data/
│   └── README.md         # Dataset layout and download notes
├── experiments/
│   └── kdd_config.yaml   # Main experiment configuration
├── results/
│   └── README.md         # Reported numbers and scripts
├── tests/
│   └── test_retriever.py
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
git clone https://github.com/wholetherapi/emails_github_1643_m2v7c1_fedkg-rag.git
cd fedkg-rag
pip install -r requirements.txt
python src/train.py --config experiments/kdd_config.yaml
```

## Citation

```bibtex
@inproceedings{carter2026federated,
  title={Federated Retrieval Augmentation over Heterogeneous Knowledge Graphs},
  author={Carter, Patrick and others},
  booktitle={Under review at KDD 2026 - ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```
