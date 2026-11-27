# -*- coding: utf-8 -*-
"""Avaliação com o script oficial da CoNLL 2018 (tools/conll18_ud_eval.py v1.2, opção -v)."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Dict

from .run_pipeline import ud_systems
from .ud_alignment import OFFICIAL_SCRIPT

logger = logging.getLogger(__name__)


def run_conll18_eval(gold_path: str, system_path: str, verbose: bool = True) -> str:
    cmd = [sys.executable, OFFICIAL_SCRIPT] + (["-v"] if verbose else []) + [gold_path, system_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"UD eval falhou.\nCMD: {' '.join(cmd)}\n"
                           f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def eval_all_ud(config: Dict) -> Dict[str, str]:
    results_dir = os.path.join(config["paths"]["results_dir"], "ud_eval")
    os.makedirs(results_dir, exist_ok=True)
    gold = config["paths"]["ud_gold_conllu"]
    outputs = config["paths"]["outputs_dir"]

    out_texts: Dict[str, str] = {}
    for name in ud_systems(config):
        sys_path = os.path.join(outputs, name, "bosque_pred_e2e.conllu")
        if not os.path.exists(sys_path):
            raise FileNotFoundError(f"[UD EVAL] arquivo do sistema não existe: {sys_path}")
        logger.info(f"[UD EVAL] avaliando {name}: {sys_path}")
        txt = run_conll18_eval(gold, sys_path)
        out_path = os.path.join(results_dir, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)
        out_texts[name] = txt
        logger.info(f"[UD EVAL] salvo: {out_path}")
    return out_texts
