# -*- coding: utf-8 -*-
"""Leitura do WikiNER (aij-wikiner-pt-wp3)."""
from __future__ import annotations

from typing import Dict, List, Optional


def load_wikiner(path: str, max_sentences: Optional[int] = None) -> List[Dict]:
    """Lê o WikiNER (uma sentença por linha; tokens 'palavra|pos|ner').

    ``max_sentences`` limita o número de LINHAS lidas (linhas vazias contam): com 5.000,
    obtêm-se as 4.701 sentenças do artigo. Devolve [{"tokens": [...], "tags": [...]}].
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break
            tokens, tags = [], []
            for tok in (p for p in line.strip().split(" ") if p):
                try:
                    word, _pos, ner = tok.split("|")
                except ValueError:
                    continue
                tokens.append(word)
                tags.append(ner)
            if tokens:
                sentences.append({"tokens": tokens, "tags": tags})
    return sentences
