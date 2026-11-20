# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from typing import Dict
from datetime import datetime

from tabulate import tabulate

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_report(config: Dict, ud_eval_texts: Dict[str, str], ner_results: Dict, im_results: Dict, versions: Dict):
    results_dir = config["paths"]["results_dir"]
    ensure_dir(results_dir)

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": config.get("seed"),
            "versions": versions
        },
        "ud_eval_official": ud_eval_texts,   # stdout oficial (texto)
        "wikiner": ner_results,
        "informedness_markedness": im_results
    }

    json_path = os.path.join(results_dir, "full_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # TXT breve (sem “recalcular” métricas UD)
    txt_path = os.path.join(results_dir, "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("RELATÓRIO - PIPELINE NLP PT-BR (spaCy/Stanza/UDPipe)\n")
        f.write("=" * 100 + "\n\n")
        f.write("1) AVALIAÇÃO UD OFICIAL (stdout do conll18_ud_eval.py)\n\n")
        for sysname, text in ud_eval_texts.items():
            f.write("-" * 80 + "\n")
            f.write(f"SISTEMA: {sysname}\n")
            f.write("-" * 80 + "\n")
            f.write(text.strip() + "\n\n")

        f.write("\n2) WIKINER (NER)\n\n")
        for sysname, r in ner_results.items():
            tl = r.get("token_level", {})
            f.write(f"- {sysname}: token-level P={tl.get('precision',0):.4f} "
                    f"R={tl.get('recall',0):.4f} F1={tl.get('f1',0):.4f}\n")
            if r.get("span_level"):
                sl = r["span_level"]
                f.write(f"  span-level(seqeval) P={sl.get('precision',0):.4f} "
                        f"R={sl.get('recall',0):.4f} F1={sl.get('f1',0):.4f}\n")

        f.write("\n3) INFORMEDNESS & MARKEDNESS (amostras controladas)\n")
        for task, table in im_results.items():
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"TASK: {task}\n")
            f.write("-" * 80 + "\n")
            rows = []
            for lab, d in sorted(table.items()):
                rows.append([lab, d["informedness"], d["markedness"], d["TPR"], d["PPV"], d["support"]])
            f.write(tabulate(rows, headers=["Label","Informedness","Markedness","TPR","PPV","Support"], tablefmt="grid", floatfmt=".4f"))
            f.write("\n")

    return {"json": json_path, "txt": txt_path}
