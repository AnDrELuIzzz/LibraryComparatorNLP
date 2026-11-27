# -*- coding: utf-8 -*-
"""Baixa os modelos do estudo (UDPipe, spaCy e Stanza). main.py faz o mesmo automaticamente.

Uso:  python scripts/download_models.py [--config config.json]
"""
import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from main import load_config  # noqa: E402
from src.resources import ensure_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    ensure_all(load_config(args.config))
    print("Modelos prontos.")


if __name__ == "__main__":
    main()
