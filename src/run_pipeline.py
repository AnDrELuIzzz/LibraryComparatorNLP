# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple, Optional

from .model_wrapper import get_model_wrapper
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_word_id(id_field: str) -> bool:
    # remove BOM se existir
    id_clean = id_field.lstrip("\ufeff")
    return id_clean.isdigit()


def _is_mwt_id(id_field: str) -> bool:
    id_clean = id_field.lstrip("\ufeff")
    if "-" not in id_clean:
        return False
    a, b = id_clean.split("-", 1)
    return a.isdigit() and b.isdigit()


def normalize_ud_tree(heads: List[int], deprels: List[str]) -> Tuple[List[int], List[str]]:
    """
    Deixa HEAD/DEPREL sempre válidos para o conll18_ud_eval.py:
    - HEAD em 0..N (0=root) e sem self-loop
    - exatamente 1 raiz
    - sem ciclos (quebra ciclos reanexando no root)
    O conll18 aborta em ciclos e múltiplas raízes. [web:25]
    """
    n = len(heads)
    if n == 0:
        return heads, deprels

    # 0) Coagir tipos e alinhar tamanhos
    new_heads: List[int] = []
    for h in heads:
        try:
            new_heads.append(int(h))
        except Exception:
            new_heads.append(0)

    new_deprels = list(deprels) if deprels is not None else ["dep"] * n
    if len(new_deprels) != n:
        new_deprels = (new_deprels + ["dep"] * n)[:n]

    # 1) Corrigir HEAD fora do range e self-loop
    for i, h in enumerate(new_heads):
        wid = i + 1
        if h < 0 or h > n:
            new_heads[i] = 0
            new_deprels[i] = "root"
        elif h == wid:
            new_heads[i] = 0
            new_deprels[i] = "root"

    # 2) Garantir pelo menos uma raiz (para escolher root_id)
    roots = [i for i, h in enumerate(new_heads) if h == 0]
    if not roots:
        new_heads[0] = 0
        new_deprels[0] = "root"
        roots = [0]

    root_id = roots[0] + 1  # 1-indexed

    # 3) Quebrar ciclos: caminhada de pais; ao detectar ciclo, reanexa no root
    for start in range(1, n + 1):
        seen = set()
        cur = start
        while True:
            h = new_heads[cur - 1]
            if h == 0:
                break
            if h < 0 or h > n:
                new_heads[cur - 1] = root_id if cur != root_id else 0
                new_deprels[cur - 1] = "dep" if cur != root_id else "root"
                break
            if h == cur or h in seen:
                new_heads[cur - 1] = root_id if cur != root_id else 0
                new_deprels[cur - 1] = "dep" if cur != root_id else "root"
                break
            seen.add(cur)
            cur = h

    # 4) Garantir exatamente 1 raiz (pode ter surgido mais de uma após correções)
    roots = [i for i, h in enumerate(new_heads) if h == 0]
    if len(roots) == 0:
        new_heads[root_id - 1] = 0
        new_deprels[root_id - 1] = "root"
        roots = [root_id - 1]

    if len(roots) > 1:
        main_root = roots[0]
        main_root_id = main_root + 1
        new_deprels[main_root] = "root"
        for r in roots[1:]:
            # reanexa as "raízes extras" na raiz principal
            new_heads[r] = main_root_id
            new_deprels[r] = "dep"

    return new_heads, new_deprels
    """
    Normaliza árvore UD para não quebrar o conll18_ud_eval.py:
    - HEAD em 0..N (0=root)
    - exatamente 1 raiz por sentença
    """
    n = len(heads)
    if n == 0:
        return heads, deprels

    new_heads: List[int] = []
    for h in heads:
        try:
            new_heads.append(int(h))
        except Exception:
            new_heads.append(0)

    new_deprels = list(deprels) if deprels is not None else ["dep"] * n
    if len(new_deprels) != n:
        new_deprels = (new_deprels + ["dep"] * n)[:n]

    # HEAD fora do range -> reanexa no token 1 (temporário)
    for i, h in enumerate(new_heads):
        if h < 0 or h > n:
            new_heads[i] = 1
            new_deprels[i] = "dep"

    roots = [i for i, h in enumerate(new_heads) if h == 0]
    if len(roots) == 0:
        new_heads[0] = 0
        new_deprels[0] = "root"
    elif len(roots) > 1:
        root = roots[0]
        new_deprels[root] = "root"
        for i in roots[1:]:
            new_heads[i] = root + 1
            new_deprels[i] = "dep"
    else:
        new_deprels[roots[0]] = "root"
    


    return new_heads, new_deprels


def build_pred_conllu_aligned_to_gold(
    framework: str,
    model_name: str,
    gold_conllu_path: str,
    output_path: str,
):
    """
    SYSTEM.conllu com tokenização/segmentação IDÊNTICA ao GOLD:
    - Copia FORM e MISC do GOLD (garante concatenação igual).
    - Substitui LEMMA/UPOS/HEAD/DEPREL pelas predições feitas sobre os tokens gold.
    """
    wrapper = get_model_wrapper(framework, model_name)
    wrapper.load_model()

    def process_block(block_lines: List[str], out_f):
        # Coletar tokens das "word lines" (IDs inteiros), preservando ordem original
        word_lines_idx: List[int] = []
        word_cols: List[List[str]] = []
        gold_tokens: List[str] = []

        for i, line in enumerate(block_lines):
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if _is_word_id(cols[0]):
                word_lines_idx.append(i)
                word_cols.append(cols)
                gold_tokens.append(cols[1])

        # Se não tem palavras, só escreve o bloco como está
        if not gold_tokens:
            for line in block_lines:
                out_f.write(line + "\n")
            out_f.write("\n")
            return

        # Predições em cima dos tokens gold
        pred_upos = list(wrapper.pos_tag_gold(gold_tokens))
        pred_lemmas = list(wrapper.lemmatize_gold(gold_tokens))
        pred_heads, pred_deprels = wrapper.dependency_parse_gold(gold_tokens)
        pred_heads = list(pred_heads)
        pred_deprels = list(pred_deprels)

        n = len(gold_tokens)
        if len(pred_upos) != n:
            pred_upos = (pred_upos + ["_"] * n)[:n]
        if len(pred_lemmas) != n:
            pred_lemmas = (pred_lemmas + ["_"] * n)[:n]
        if len(pred_heads) != n:
            pred_heads = (pred_heads + [0] * n)[:n]
        if len(pred_deprels) != n:
            pred_deprels = (pred_deprels + ["dep"] * n)[:n]

        pred_heads, pred_deprels = normalize_ud_tree(pred_heads, pred_deprels)

        # Montar versões “substituídas” das word lines
        replaced = {}
        for k in range(n):
            cols = word_cols[k][:]
            # cols: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            # PRESERVA: ID, FORM, MISC (não mexe em cols[0], cols[1], cols[9])
            cols[2] = pred_lemmas[k] or "_"
            cols[3] = pred_upos[k] or "_"
            cols[4] = "_"   # XPOS
            cols[5] = "_"   # FEATS
            cols[6] = str(int(pred_heads[k])) if pred_heads[k] is not None else "_"
            cols[7] = pred_deprels[k] or "_"
            cols[8] = "_"   # DEPS
            replaced[word_lines_idx[k]] = "\t".join(cols)

        # Escrever o bloco na MESMA ordem do GOLD (in-place)
        for i, line in enumerate(block_lines):
            if i in replaced:
                out_f.write(replaced[i] + "\n")
            else:
                out_f.write(line + "\n")
        out_f.write("\n")

    # Ler por blocos (sentenças) e escrever
    with open(gold_conllu_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        block: List[str] = []
        for raw in f_in:
            line = raw.rstrip("\n")
            if line.strip() == "":
                if block:
                    process_block(block, f_out)
                    block = []
            else:
                block.append(line)
        if block:
            process_block(block, f_out)

    logger.info(f"[{framework}] CoNLL-U aligned-to-gold salvo em: {output_path}")


def build_pred_wikiner_bio(
    framework: str,
    model_name: str,
    wikiner_path: str,
    output_path: str,
    max_sentences: int = 5000,
):
    loader = DataLoader(conllu_path="(unused)", wikiner_path=wikiner_path)
    samples = loader.load_wikiner(max_sentences=max_sentences)

    wrapper = get_model_wrapper(framework, model_name)
    wrapper.load_model()

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            tokens = s["tokens"]
            tags = wrapper.ner_gold(tokens)
            f.write(" ".join([f"{t}|{y}" for t, y in zip(tokens, tags)]) + "\n")

    logger.info(f"[{framework}] WikiNER predito salvo em: {output_path}")


def run_all_pipelines(config: Dict):
    outputs_dir = config["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)

    gold_path = config["paths"]["ud_gold_conllu"]
    wikiner_path = config["paths"]["wikiner_path"]
    max_ner = config.get("wikiner", {}).get("max_sentences", 5000)

    # spaCy
    spacy_out = os.path.join(outputs_dir, "spacy")
    ensure_dir(spacy_out)
    build_pred_conllu_aligned_to_gold(
        "spacy",
        config["models"]["spacy"],
        gold_path,
        os.path.join(spacy_out, "bosque_pred.conllu"),
    )
    build_pred_wikiner_bio(
        "spacy",
        config["models"]["spacy"],
        wikiner_path,
        os.path.join(spacy_out, "wikiner_pred.bio"),
        max_sentences=max_ner,
    )

    # Stanza
    stanza_out = os.path.join(outputs_dir, "stanza")
    ensure_dir(stanza_out)
    build_pred_conllu_aligned_to_gold(
        "stanza",
        config["models"]["stanza_lang"],
        gold_path,
        os.path.join(stanza_out, "bosque_pred.conllu"),
    )
    build_pred_wikiner_bio(
        "stanza",
        config["models"]["stanza_lang"],
        wikiner_path,
        os.path.join(stanza_out, "wikiner_pred.bio"),
        max_sentences=max_ner,
    )

    # UDPipe
    udpipe_out = os.path.join(outputs_dir, "udpipe")
    ensure_dir(udpipe_out)
    build_pred_conllu_aligned_to_gold(
        "udpipe",
        config["paths"]["udpipe_model_path"],
        gold_path,
        os.path.join(udpipe_out, "bosque_pred.conllu"),
    )
    build_pred_wikiner_bio(
        "udpipe",
        config["paths"]["udpipe_model_path"],
        wikiner_path,
        os.path.join(udpipe_out, "wikiner_pred.bio"),
        max_sentences=max_ner,
    )

    logger.info("Todas as pipelines foram executadas com sucesso!")
