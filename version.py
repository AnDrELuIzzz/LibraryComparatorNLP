"""
Version and Configuration Module
Centraliza versões de bibliotecas e modelos.
"""

import importlib
import sys
from datetime import datetime

__version__ = "0.3.0"
__author__ = "André Luiz Araujo Carvalho, Manoel Guaranha"
__project__ = "Comparação de Ferramentas de PLN - PT-BR"

PINNED_VERSIONS = {
    "torch": "2.1.2",
    "transformers": "4.35.2",
    "scikit-learn": "1.3.2",
    "stanza": "1.8.0",
    "spacy": "3.7.2",
    "ufal.udpipe": "1.3.0",
    "pandas": "2.1.3",
    "numpy": "1.26.2",
    "scipy": "1.11.4",
    "statsmodels": "0.14.0"
}

MODELS = {
    "huggingface_ner": {
        "model": "pierreguillou/bert-base-cased-pt-lenerbr",
        "version": "1.0"
    },
    "stanza": {
        "model": "pt",
        "processors": "tokenize,pos,lemma,depparse,ner",
        "version": "1.8.0"
    },
    "spacy": {
        "model": "pt_core_news_sm",
        "version": "3.7.0"
    },
    "udpipe": {
        "model": "portuguese-bosque-ud-2.5-191206.udpipe",
        "version": "2.5"
    }
}

TIMESTAMP = datetime.now().isoformat()


def get_package_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


def print_versions() -> dict:
    info = {
        "python": sys.version,
        "project_version": __version__,
        "timestamp": TIMESTAMP,
        "packages": {},
        "models": MODELS
    }
    print("=" * 70)
    print(f"{__project__} v{__version__}")
    print("=" * 70)
    print("Python:", sys.version)
    print("Timestamp:", TIMESTAMP)
    print("\nPackages:")
    for pkg, expected in PINNED_VERSIONS.items():
        installed = get_package_version(pkg)
        status = "OK" if installed == expected else "WARN"
        print(f"{status} {pkg}: {installed} (expected {expected})")
        info["packages"][pkg] = installed
    print("\nModels:")
    for k, v in MODELS.items():
        print(f"{k}: {v}")
    print("=" * 70)
    return info


if __name__ == "__main__":
    print_versions()
