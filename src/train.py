import argparse

import yaml

from model import FederatedRAG
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = FederatedRAG(**cfg["model"])
    print("training federated retriever on", cfg["datasets"])


if __name__ == "__main__":
    main()
