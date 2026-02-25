# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import subprocess
import urllib.request
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def ensure_conll18_script(tools_dir: str, url: str) -> str:
    ensure_dir(tools_dir)
    script_path = os.path.join(tools_dir, "conll18_ud_eval.py")
    if not os.path.exists(script_path):
        logger.info(f"Baixando conll18_ud_eval.py de {url} ...")
        urllib.request.urlretrieve(url, script_path)
    return script_path


def run_conll18_eval(
    script_path: str,
    gold_path: str,
    system_path: str,
    verbose: bool = True
) -> str:
    cmd = [sys.executable, script_path]
    if verbose:
        cmd.append("-v")
    cmd += [gold_path, system_path]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "UD eval falhou.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout


def eval_all_ud(config: Dict) -> Dict[str, str]:
    tools_dir = config["paths"]["tools_dir"]
    results_dir = os.path.join(config["paths"]["results_dir"], "ud_eval")
    ensure_dir(results_dir)

    script_path = ensure_conll18_script(
        tools_dir=tools_dir,
        url=config["ud_eval"]["download_url"],
    )

    gold = config["paths"]["ud_gold_conllu"]
    outputs = config["paths"]["outputs_dir"]
    verbose = bool(config["ud_eval"].get("verbose", True))

    systems = {
        "spacy": os.path.join(outputs, "spacy", "bosque_pred_e2e.conllu"),
        "stanza": os.path.join(outputs, "stanza", "bosque_pred_e2e.conllu"),
        "udpipe": os.path.join(outputs, "udpipe", "bosque_pred_e2e.conllu"),
    }

    out_texts: Dict[str, str] = {}
    for name, sys_path in systems.items():
        if not os.path.exists(sys_path):
            raise FileNotFoundError(f"[UD EVAL] arquivo do sistema não existe: {sys_path}")

        logger.info(f"[UD EVAL] avaliando {name}: {sys_path}")
        txt = run_conll18_eval(script_path, gold, sys_path, verbose=verbose)

        out_path = os.path.join(results_dir, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

        out_texts[name] = txt
        logger.info(f"[UD EVAL] salvo: {out_path}")

    return out_texts
