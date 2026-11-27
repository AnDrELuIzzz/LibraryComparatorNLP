# -*- coding: utf-8 -*-
"""Versões do ambiente, registradas nos metadados de execução (results/full_results.json)."""
import importlib.metadata
import platform
import sys

PACKAGES = ("spacy", "pt_core_news_lg", "stanza", "torch", "ufal.udpipe", "seqeval",
            "scikit-learn", "numpy")


def get_versions() -> dict:
    versions = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions
