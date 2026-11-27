# -*- coding: utf-8 -*-
"""Relatório consolidado do pipeline: results/full_results.json e results/report.txt."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict


def save_report(config: Dict, ud_eval_texts: Dict[str, str], ner_results: Dict, versions: Dict) -> Dict[str, str]:
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    payload = {
        "metadata": {"timestamp": datetime.now().isoformat(), "seed": config.get("seed"), "versions": versions},
        "ud_eval_official": ud_eval_texts,  # stdout do conll18_ud_eval.py -v
        "wikiner": ner_results,
    }
    json_path = os.path.join(results_dir, "full_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(results_dir, "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("RELATÓRIO - PIPELINE NLP PT-BR (spaCy/Stanza/UDPipe)\n")
        f.write("=" * 100 + "\n\n")
        f.write("Versões: " + json.dumps(versions, ensure_ascii=False) + "\n\n")
        f.write("1) AVALIAÇÃO UD OFICIAL (stdout do conll18_ud_eval.py -v)\n\n")
        for sysname, text in ud_eval_texts.items():
            f.write("-" * 80 + "\n")
            f.write(f"SISTEMA: {sysname}\n")
            f.write("-" * 80 + "\n")
            f.write(text.strip() + "\n\n")

        f.write("\n2) WIKINER (NER; só o spaCy tem modelo para o português)\n\n")
        for sysname, r in ner_results.items():
            sl = r.get("span_level") or {}
            f.write(f"- {sysname}: por entidade (seqeval) P={sl.get('precision', 0):.4f} "
                    f"R={sl.get('recall', 0):.4f} F1={sl.get('f1', 0):.4f}\n")
            tl = r.get("token_level", {})
            f.write(f"  por token (gold IOB2) P={tl.get('precision', 0):.4f} "
                    f"R={tl.get('recall', 0):.4f} F1={tl.get('f1', 0):.4f}\n")

        f.write("\n3) ANÁLISES COMPLEMENTARES (sobreposição, bootstrap, BI/MK alinhados): "
                "results/reanalysis/tabelas.md\n")

    return {"json": json_path, "txt": txt_path}
