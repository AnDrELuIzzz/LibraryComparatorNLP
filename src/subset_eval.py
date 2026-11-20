# -*- coding: utf-8 -*-
"""Métricas derivadas do alinhamento oficial: por subconjunto de sentenças,
sobre as palavras alinhadas por todas as ferramentas e com lemas sem distinção
entre maiúsculas e minúsculas.

Definições (sempre sobre palavras gold):
  recall      = acertos / palavras gold do subconjunto   (mesmo numerador e
                denominador do recall oficial, restritos ao subconjunto)
  aligned_acc = acertos / palavras gold alinhadas do subconjunto
Precisão e F1 não são definidos por subconjunto de sentenças gold, porque as
palavras preditas não pertencem a uma sentença gold de forma unívoca.
"""
from __future__ import annotations

import io
from typing import Dict, Iterable, List

from .ud_alignment import KEYS, AlignedWord, ce, gold_word_sent_ids

METRICS = tuple(KEYS)  # UPOS, UFeats, Lemmas, Lemmas_ci, UAS, LAS


def subset_scores(pairs: List[AlignedWord], word_sent: List[str], ids: Iterable[str]) -> Dict:
    ids = set(ids)
    gold_n = sum(1 for s in word_sent if s in ids)
    sel = [p for p in pairs if p.sent_id in ids]
    out = {"n_sentences": len(ids), "gold_words": gold_n, "aligned_words": len(sel)}
    for k in METRICS:
        c = sum(p.correct[k] for p in sel)
        out[k] = {"recall": 100 * c / gold_n if gold_n else None,
                  "aligned_acc": 100 * c / len(sel) if sel else None}
    return out


def intersection_accuracy(pairs_by_system: Dict[str, List[AlignedWord]]) -> Dict:
    """Acurácia restrita às palavras gold alinhadas por TODAS as ferramentas."""
    maps = {name: {p.gold_index: p for p in pairs} for name, pairs in pairs_by_system.items()}
    common = set.intersection(*(set(m) for m in maps.values()))
    out = {"common_words": len(common)}
    for name, m in maps.items():
        out[name] = {k: 100 * sum(m[i].correct[k] for i in common) / len(common) for k in METRICS}
    return out


def _lowercase_lemmas(path: str) -> io.StringIO:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) == 10 and cols[0].isdigit():
                cols[2] = cols[2].lower()
                line = "\t".join(cols) + "\n"
            lines.append(line)
    return io.StringIO("".join(lines))


def lemma_case_insensitive(gold_path: str, system_path: str) -> Dict:
    """Escores oficiais de Lemmas e BLEX com lemas em minúsculas na referência e na predição."""
    def run(gold_file, sys_file):
        return ce.evaluate(ce.load_conllu(gold_file), ce.load_conllu(sys_file))
    with open(gold_path, encoding="utf-8") as g, open(system_path, encoding="utf-8") as s:
        orig = run(g, s)
    low = run(_lowercase_lemmas(gold_path), _lowercase_lemmas(system_path))
    pct = lambda x: 100 * x
    return {
        "original": {"Lemmas_f1": pct(orig["Lemmas"].f1), "Lemmas_aligned_acc": pct(orig["Lemmas"].aligned_accuracy),
                     "BLEX_f1": pct(orig["BLEX"].f1)},
        "case_insensitive": {"Lemmas_f1": pct(low["Lemmas"].f1), "Lemmas_aligned_acc": pct(low["Lemmas"].aligned_accuracy),
                             "BLEX_f1": pct(low["BLEX"].f1)},
    }


def unexpanded_contractions(gold_path: str, system_path: str) -> Dict[str, int]:
    """Tokens multipalavra da referência (ex.: do = de + o) que a ferramenta deixou
    como palavra única, isto é, sem expansão: o span do token gold coincide com o
    span de uma única palavra predita que não pertence a um token multipalavra."""
    with open(gold_path, encoding="utf-8") as f:
        gold = ce.load_conllu(f)
    with open(system_path, encoding="utf-8") as f:
        system = ce.load_conllu(f)
    gold_mwt = {(w.span.start, w.span.end) for w in gold.words if w.is_multiword}
    single = {(w.span.start, w.span.end) for w in system.words if not w.is_multiword}
    return {"gold_multiword_tokens": len(gold_mwt), "unexpanded": len(gold_mwt & single)}


def mwt_count(system_path: str) -> int:
    """Nº de tokens multipalavra (linhas com intervalo, ex. '3-4') no CoNLL-U predito."""
    n = 0
    with open(system_path, encoding="utf-8") as f:
        for line in f:
            first = line.split("\t", 1)[0]
            if "-" in first and first.replace("-", "").isdigit():
                n += 1
    return n


__all__ = ["subset_scores", "intersection_accuracy", "lemma_case_insensitive", "mwt_count", "unexpanded_contractions",
           "gold_word_sent_ids", "METRICS"]
