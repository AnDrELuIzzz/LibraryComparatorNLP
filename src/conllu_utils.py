# -*- coding: utf-8 -*-
"""
Utilitários CoNLL-U:
- Ler CoNLL-U (inclui MISC e SpaceAfter=No)
- Detokenizar texto a partir de tokens + MISC
- Escrever CoNLL-U a partir de estruturas preditas
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re

@dataclass
class ConlluToken:
    id: str
    form: str
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: str = "_"
    head: str = "_"
    deprel: str = "_"
    deps: str = "_"
    misc: str = "_"

@dataclass
class ConlluSentence:
    sent_id: str
    text: str
    tokens: List[ConlluToken]

_SPACEAFTER_RE = re.compile(r"(?:^|\|)SpaceAfter=No(?:\||$)")

def misc_has_spaceafter_no(misc: str) -> bool:
    if not misc or misc == "_":
        return False
    return _SPACEAFTER_RE.search(misc) is not None

def detokenize_from_tokens(tokens: List[ConlluToken]) -> str:
    """
    Reconstrói texto usando FORM e SpaceAfter=No no MISC.
    """
    out = []
    for i, tok in enumerate(tokens):
        out.append(tok.form)
        if i < len(tokens) - 1:
            if not misc_has_spaceafter_no(tok.misc):
                out.append(" ")
    return "".join(out)

def read_conllu(path: str) -> List[ConlluSentence]:
    """
    Lê CoNLL-U preservando sent_id, text, tokens e MISC.
    Ignora linhas de multi-word tokens (IDs com '-').
    """
    sentences: List[ConlluSentence] = []
    cur_sent_id = ""
    cur_text = ""
    cur_tokens: List[ConlluToken] = []

    def flush():
        nonlocal cur_sent_id, cur_text, cur_tokens
        if cur_tokens:
            if not cur_sent_id:
                cur_sent_id = f"sent_{len(sentences)+1}"
            if not cur_text:
                cur_text = detokenize_from_tokens(cur_tokens)
            sentences.append(ConlluSentence(cur_sent_id, cur_text, cur_tokens))
        cur_sent_id, cur_text, cur_tokens = "", "", []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("# sent_id"):
                cur_sent_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("# text"):
                cur_text = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if "-" in cols[0] or "." in cols[0]:
                # MWT / empty nodes ignorados aqui
                continue

            tok = ConlluToken(
                id=cols[0],
                form=cols[1],
                lemma=cols[2],
                upos=cols[3],
                xpos=cols[4],
                feats=cols[5],
                head=cols[6],
                deprel=cols[7],
                deps=cols[8],
                misc=cols[9],
            )
            cur_tokens.append(tok)

    flush()
    return sentences

def build_document_text_from_gold(gold_sents: List[ConlluSentence]) -> str:
    """
    Concatena textos gold para formar um "documento" único (separado por newline).
    Newlines são tratados como whitespace pelos tokenizers.
    """
    return "\n".join([s.text for s in gold_sents])

def compute_offsets_by_greedy_search(text: str, tokens: List[str], start_pos: int = 0) -> List[Tuple[int, int]]:
    """
    Encontra offsets (start,end) dos tokens em 'text' por busca gulosa.
    É robusto para muitos casos práticos quando o texto veio do próprio gold.
    """
    offsets = []
    pos = start_pos
    for t in tokens:
        idx = text.find(t, pos)
        if idx < 0:
            # fallback: tenta ignorar espaços
            stripped = t.strip()
            idx = text.find(stripped, pos)
            if idx < 0:
                # último recurso: ancora no pos atual
                idx = pos
                end = idx + len(t)
                offsets.append((idx, end))
                pos = end
                continue
            t = stripped
        end = idx + len(t)
        offsets.append((idx, end))
        pos = end
    return offsets

def misc_spaceafter_from_offsets(offsets: List[Tuple[int, int]]) -> List[str]:
    """
    Deriva MISC = SpaceAfter=No se next_start == cur_end.
    """
    misc = ["_"] * len(offsets)
    for i in range(len(offsets) - 1):
        cur_end = offsets[i][1]
        next_start = offsets[i + 1][0]
        if next_start == cur_end:
            misc[i] = "SpaceAfter=No"
    return misc

def _clean_field(x: str) -> str:
    if x is None:
        return "_"
    s = str(x)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = s.strip()
    return s if s else "_"

def _clean_form(x: str) -> str:
    s = _clean_field(x)
    # FORM não pode conter whitespace; se virar "_" por ser vazio/whitespace, usa um placeholder
    if s == "_":
        return "WS"
    # garante sem espaços
    s = s.replace(" ", "")
    return s if s else "WS"

# ✅ CORREÇÃO: removido o segundo bloco "with open(...)" duplicado que sobrescrevia
# o primeiro sem aplicar _clean_field/_clean_form, permitindo \n e \t nos campos.
def write_conllu(path: str, sentences: List[ConlluSentence]):
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            sent_id = _clean_field(sent.sent_id)
            text = _clean_field(sent.text)  # garante 1 linha (remove \n embutidos)

            f.write(f"# sent_id = {sent_id}\n")
            f.write(f"# text = {text}\n")

            for tok in sent.tokens:
                f.write("\t".join([
                    _clean_field(tok.id),
                    _clean_form(tok.form),   # \n/\t → "WS" se só whitespace
                    _clean_field(tok.lemma),
                    _clean_field(tok.upos),
                    _clean_field(tok.xpos),
                    _clean_field(tok.feats),
                    _clean_field(tok.head),
                    _clean_field(tok.deprel),
                    _clean_field(tok.deps),
                    _clean_field(tok.misc),
                ]) + "\n")
            f.write("\n")

def reconstruct_text_from_gold(tokens: List[str], miscs: List[str]) -> str:
    """
    Reconstrói texto gold respeitando SpaceAfter=No do MISC.
    Exemplo: ["d'", "água"] + ["SpaceAfter=No"] → "d'água"
    """
    text_parts = []
    for tok, misc in zip(tokens, miscs):
        text_parts.append(tok)
        if "SpaceAfter=No" not in misc:
            text_parts.append(" ")
    return "".join(text_parts).rstrip()
