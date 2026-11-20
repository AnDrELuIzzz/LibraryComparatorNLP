# -*- coding: utf-8 -*-
"""Sobreposição entre o teste do Bosque e as partições de versões anteriores do UD.

O teste usado no estudo (data/pt_bosque-ud-test.conllu) é o da versão 2.12 do UD
(idêntico nas versões 2.12 a 2.17). O UDPipe (modelo UD 2.5) e o spaCy
(pt_core_news_lg, Bosque v2.8) foram treinados em versões anteriores, cujas
partições são diferentes, pois o Bosque foi reparticionado na versão 2.10.

O arquivo data/bosque_test_partition_origin.tsv registra, para cada sent_id do
teste, a variedade (CF = CETENFolha/PB; CP = CETEMPúblico/PE) e a partição de
origem em cada versão. Ele é gerado por ``build_origin_table`` a partir das
partições oficiais (baixadas com scripts/download_ud_bosque.py) e versionado
para que a reanálise rode sem acesso à rede.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN_TSV = os.path.join(ROOT, "data", "bosque_test_partition_origin.tsv")
RELEASES = ("2.5", "2.8", "2.12", "2.15")
SPLITS = ("train", "dev", "test")


def sent_ids(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.split("=", 1)[1].strip() for l in f if l.startswith("# sent_id")]


def build_origin_table(test_path: str, releases_dir: str, out_path: str = ORIGIN_TSV) -> List[Dict]:
    """Cruza os sent_id do teste com train/dev/test de cada versão (arquivos split_rX.conllu)."""
    membership = {}
    for rel in RELEASES:
        for split in SPLITS:
            path = os.path.join(releases_dir, f"{split}_r{rel}.conllu")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} ausente: rode scripts/download_ud_bosque.py")
            for sid in sent_ids(path):
                membership.setdefault(rel, {})[sid] = split
    rows = []
    for sid in sent_ids(test_path):
        row = {"sent_id": sid, "variety": "PB" if sid.startswith("CF") else "PE"}
        for rel in RELEASES:
            row[f"split_{rel}"] = membership[rel].get(sid, "absent")
        rows.append(row)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return rows


def load_origin_table(path: str = ORIGIN_TSV) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return {r["sent_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def subsets(origin: Dict[str, Dict[str, str]], release: str = "2.5") -> Dict[str, set]:
    """Subconjuntos de sentenças do teste segundo a partição de origem em ``release``."""
    col = f"split_{release}"
    all_ids = set(origin)
    return {
        "total": all_ids,
        "treino": {s for s in all_ids if origin[s][col] == "train"},
        "desenvolvimento": {s for s in all_ids if origin[s][col] == "dev"},
        "ineditas": {s for s in all_ids if origin[s][col] == "test"},
        "fora_do_treino": {s for s in all_ids if origin[s][col] != "train"},
        "PB": {s for s in all_ids if origin[s]["variety"] == "PB"},
        "PE": {s for s in all_ids if origin[s]["variety"] == "PE"},
    }


def overlap_summary(origin: Dict[str, Dict[str, str]]) -> Dict:
    n = len(origin)
    out = {"n_sentences": n, "by_release": {}}
    for rel in RELEASES:
        counts = {}
        for r in origin.values():
            counts[r[f"split_{rel}"]] = counts.get(r[f"split_{rel}"], 0) + 1
        out["by_release"][rel] = {k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in sorted(counts.items())}
    var = {}
    for r in origin.values():
        key = f"{r['variety']}_{r['split_2.5']}"
        var[key] = var.get(key, 0) + 1
    out["variety_by_split_2.5"] = dict(sorted(var.items()))
    return out
