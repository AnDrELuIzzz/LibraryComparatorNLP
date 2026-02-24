# -*- coding: utf-8 -*-
import platform
import sys

def print_versions() -> dict:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    # libs opcionais
    for pkg in ["numpy", "pandas", "spacy", "stanza", "ufal.udpipe", "sklearn", "scipy"]:
        try:
            mod = __import__(pkg.split(".")[0])
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not_installed"

    return versions
