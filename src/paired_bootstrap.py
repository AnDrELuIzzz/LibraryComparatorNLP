# -*- coding: utf-8 -*-
"""Bootstrap pareado não paramétrico sobre sentenças (Koehn, 2004; Berg-Kirkpatrick et al., 2012).

Para um subconjunto de sentenças gold, reamostra as sentenças com reposição,
com os MESMOS índices para as duas ferramentas comparadas, e recalcula em cada
réplica o recall = soma(acertos) / soma(palavras gold), isto é, a razão das
somas (não a média de escores por sentença). Reporta o IC por percentis da
diferença A − B (Dror et al., 2018).
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .ud_alignment import AlignedWord

DEFAULT_B = 10_000
DEFAULT_SEED = 42


def per_sentence_counts(pairs: List[AlignedWord], metric: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in pairs:
        if p.correct[metric]:
            out[p.sent_id] = out.get(p.sent_id, 0) + 1
    return out


def gold_words_per_sentence(word_sent: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in word_sent:
        out[s] = out.get(s, 0) + 1
    return out


def paired_bootstrap(pairs_a: List[AlignedWord], pairs_b: List[AlignedWord], word_sent: List[str],
                     ids: Iterable[str], metric: str, B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> Dict:
    wanted = set(ids)
    ids = [s for s in dict.fromkeys(word_sent) if s in wanted]  # ordem do arquivo gold
    gold = gold_words_per_sentence(word_sent)
    ca, cb = per_sentence_counts(pairs_a, metric), per_sentence_counts(pairs_b, metric)
    n = np.array([gold[s] for s in ids], dtype=float)
    a = np.array([ca.get(s, 0) for s in ids], dtype=float)
    b = np.array([cb.get(s, 0) for s in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    ra = a[idx].sum(axis=1) / n[idx].sum(axis=1)
    rb = b[idx].sum(axis=1) / n[idx].sum(axis=1)
    d = ra - rb
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "metric": metric, "n_sentences": len(ids), "B": B, "seed": seed,
        "a": 100 * a.sum() / n.sum(), "b": 100 * b.sum() / n.sum(),
        "diff": 100 * (a.sum() - b.sum()) / n.sum(),
        "ci95": [100 * lo, 100 * hi],
    }
