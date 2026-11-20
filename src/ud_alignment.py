# -*- coding: utf-8 -*-
"""Acesso ao alinhamento de palavras do script oficial conll18_ud_eval.py (v1.2).

O script oficial calcula o alinhamento entre palavras de referência (gold) e
preditas dentro de ``evaluate()``, mas não o devolve. Para reutilizá-lo SEM
alterar o arquivo oficial (``tools/conll18_ud_eval.py`` permanece idêntico ao
distribuído pela CoNLL 2018), este módulo registra um perfilador temporário
(``sys.setprofile``) e lê a variável local ``alignment`` no retorno de
``evaluate()``. As contagens derivadas desse alinhamento são conferidas contra
os escores do próprio script (função ``check_against_official``).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_SCRIPT = os.path.join(ROOT, "tools", "conll18_ud_eval.py")
# md5 do conll18_ud_eval.py v1.2 versionado no repositório
OFFICIAL_SCRIPT_MD5 = "9532fb0844a572429361b1dba69d8ec0"


def load_official(path: str = OFFICIAL_SCRIPT):
    """Carrega o script oficial como módulo Python, sem modificá-lo."""
    spec = importlib.util.spec_from_file_location("conll18_ud_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ce = load_official()


def evaluate_with_alignment(gold_path: str, system_path: str):
    """Executa ``evaluate()`` oficial e devolve (escores, alinhamento, gold_ud).

    O alinhamento é o objeto ``Alignment`` criado pelo próprio script, com
    ``matched_words`` (pares gold/predito) e ``matched_words_map``.
    """
    with open(gold_path, encoding="utf-8") as f:
        gold_ud = ce.load_conllu(f)
    with open(system_path, encoding="utf-8") as f:
        system_ud = ce.load_conllu(f)

    captured = {}
    target = ce.evaluate.__code__

    def profiler(frame, event, arg):
        if event == "return" and frame.f_code is target:
            captured["alignment"] = frame.f_locals.get("alignment")

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        scores = ce.evaluate(gold_ud, system_ud)
    finally:
        sys.setprofile(previous)

    alignment = captured.get("alignment")
    if alignment is None:
        raise RuntimeError("Não foi possível capturar o alinhamento do conll18_ud_eval.py")
    return scores, alignment, gold_ud


# Funções de comparação idênticas às do script oficial (UPOS, UFeats, Lemmas, UAS, LAS)
def _gold_head(word):
    return word


def _system_head_factory(alignment):
    def system_head(word):
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None
    return system_head


KEYS: Dict[str, Callable] = {
    "UPOS": lambda w, head: w.columns[ce.UPOS],
    "UFeats": lambda w, head: w.columns[ce.FEATS],
    # mesma regra do script oficial: lema "_" na referência não é avaliado
    "Lemmas": lambda w, head: w.columns[ce.LEMMA] if head(w).columns[ce.LEMMA] != "_" else "_",
    "Lemmas_ci": lambda w, head: w.columns[ce.LEMMA].lower() if head(w).columns[ce.LEMMA] != "_" else "_",
    "UAS": lambda w, head: head(w.parent),
    "LAS": lambda w, head: (head(w.parent), w.columns[ce.DEPREL]),
}


@dataclass
class AlignedWord:
    gold_index: int          # posição da palavra gold em gold_ud.words
    sent_id: str             # sent_id da sentença gold
    correct: Dict[str, bool] # acerto por métrica
    gold_upos: str
    system_upos: str


def gold_word_sent_ids(gold_path: str) -> List[str]:
    """sent_id de cada palavra sintática gold, na ordem de gold_ud.words."""
    out, sid = [], None
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line.strip() and not line.startswith("#"):
                if line.split("\t")[0].isdigit():
                    out.append(sid)
    return out


def aligned_words(gold_path: str, system_path: str) -> Tuple[List[AlignedWord], int, Dict]:
    """Pares alinhados com acerto por métrica. Devolve (pares, nº de palavras gold, escores oficiais)."""
    scores, alignment, gold_ud = evaluate_with_alignment(gold_path, system_path)
    sent_of = gold_word_sent_ids(gold_path)
    if len(sent_of) != len(gold_ud.words):
        raise ValueError("Número de palavras gold diverge do script oficial")
    index = {id(w): i for i, w in enumerate(gold_ud.words)}
    system_head = _system_head_factory(alignment)
    pairs = []
    for m in alignment.matched_words:
        gi = index[id(m.gold_word)]
        correct = {k: fn(m.gold_word, _gold_head) == fn(m.system_word, system_head) for k, fn in KEYS.items()}
        pairs.append(AlignedWord(gi, sent_of[gi], correct,
                                 m.gold_word.columns[ce.UPOS], m.system_word.columns[ce.UPOS]))
    check_against_official(pairs, scores)
    return pairs, len(gold_ud.words), scores


def check_against_official(pairs: List[AlignedWord], scores: Dict) -> None:
    """Garante que as contagens derivadas do alinhamento reproduzem o script oficial."""
    for key in ("UPOS", "UFeats", "Lemmas", "UAS", "LAS"):
        mine = sum(p.correct[key] for p in pairs)
        if mine != scores[key].correct:
            raise AssertionError(f"{key}: {mine} acertos != {scores[key].correct} do script oficial")
