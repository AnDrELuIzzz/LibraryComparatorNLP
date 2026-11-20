# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple, Optional

from .model_wrapper import get_model_wrapper
from .data_loader import DataLoader
from .conllu_utils import (
    read_conllu,
    build_document_text_from_gold,
    ConlluSentence,
    ConlluToken,
    write_conllu,
    compute_offsets_by_greedy_search,
    misc_spaceafter_from_offsets,
)

logger = logging.getLogger(__name__)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_ud_tree(heads, deprels):
    """
    Normaliza árvore UD para não quebrar o conll18_ud_eval.py:
    - HEAD em 0..N (0=root), sem self-loop
    - exatamente 1 raiz
    - tenta quebrar ciclos reanexando no root
    """
    n = len(heads)
    if n == 0:
        return [], []

    new_heads = []
    for h in heads:
        try:
            new_heads.append(int(h))
        except Exception:
            new_heads.append(0)

    new_deprels = list(deprels) if deprels is not None else ["dep"] * n
    if len(new_deprels) != n:
        new_deprels = (new_deprels + ["dep"] * n)[:n]

    for i, h in enumerate(new_heads):
        wid = i + 1
        if h < 0 or h > n or h == wid:
            new_heads[i] = 0
            new_deprels[i] = "root"

    roots = [i for i, h in enumerate(new_heads) if h == 0]
    if not roots:
        new_heads[0] = 0
        new_deprels[0] = "root"
        roots = [0]

    main_root = roots[0]
    main_root_id = main_root + 1
    new_deprels[main_root] = "root"

    for r in roots[1:]:
        new_heads[r] = main_root_id
        new_deprels[r] = "dep"

    for start in range(1, n + 1):
        seen = set()
        cur = start
        while True:
            h = new_heads[cur - 1]
            if h == 0:
                break
            if h < 0 or h > n or h == cur or h in seen:
                new_heads[cur - 1] = main_root_id if (cur != main_root_id) else 0
                if cur - 1 != main_root:
                    new_deprels[cur - 1] = "dep"
                break
            seen.add(cur)
            cur = h

    roots = [i for i, h in enumerate(new_heads) if h == 0]
    if not roots:
        new_heads[main_root] = 0
        new_deprels[main_root] = "root"
        roots = [main_root]

    if len(roots) > 1:
        main_root = roots[0]
        main_root_id = main_root + 1
        new_deprels[main_root] = "root"
        for r in roots[1:]:
            new_heads[r] = main_root_id
            new_deprels[r] = "dep"

    return new_heads, new_deprels


def build_pred_conllu_end2end(
    framework: str,
    model_name: str,
    gold_conllu_path: str,
    output_path: str,
):
    """
    SYSTEM.conllu end-to-end:
    - Entrada: texto cru (concatenação dos # text do GOLD com newline)
    - Saída: tokens/lemmas/upos/heads/deprels do parse_raw (tokenização real do sistema)
    - MISC: deriva SpaceAfter=No via offsets (se existirem) ou greedy search no texto da sentença
    - MWT: insere linhas de superfície (ex: "1-2  pelo  ...") quando o modelo as produz,
      necessário para que conll18_ud_eval.py valide a concatenação de tokens corretamente.
    """
    gold_sents = read_conllu(gold_conllu_path)
    doc_text = build_document_text_from_gold(gold_sents)

    wrapper = get_model_wrapper(framework, model_name)
    wrapper.load_model()

    pred_sents = wrapper.parse_raw(doc_text)

    out: List[ConlluSentence] = []
    for i, s in enumerate(pred_sents, start=1):
        tokens: List[str] = list(s.get("tokens") or [])
        if not tokens:
            continue

        lemmas: List[str] = list(s.get("lemmas") or ["_"] * len(tokens))
        upos: List[str] = list(s.get("upos") or ["_"] * len(tokens))
        feats: List[str] = list(s.get("feats") or ["_"] * len(tokens))
        heads: List[int] = list(s.get("heads") or [0] * len(tokens))
        deprels: List[str] = list(s.get("deprels") or ["dep"] * len(tokens))

        n = len(tokens)
        if len(lemmas) != n:
            lemmas = (lemmas + ["_"] * n)[:n]
        if len(upos) != n:
            upos = (upos + ["_"] * n)[:n]
        if len(feats) != n:
            feats = (feats + ["_"] * n)[:n]
        if len(heads) != n:
            heads = (heads + [0] * n)[:n]
        if len(deprels) != n:
            deprels = (deprels + ["dep"] * n)[:n]

        heads, deprels = normalize_ud_tree(heads, deprels)

        offsets = s.get("offsets", None)
        sent_start = s.get("sent_start", None)
        sent_end = s.get("sent_end", None)

        if isinstance(sent_start, int) and isinstance(sent_end, int) and 0 <= sent_start <= sent_end <= len(doc_text):
            sent_text = doc_text[sent_start:sent_end]
        else:
            sent_text = s.get("text") or " ".join(tokens)

        if offsets and isinstance(offsets, list) and len(offsets) == n:
            misc = misc_spaceafter_from_offsets(offsets)
        else:
            off = compute_offsets_by_greedy_search(sent_text, tokens, start_pos=0)
            misc = misc_spaceafter_from_offsets(off)

        # Monta tokens regulares (1-based)
        conllu_tokens: List[ConlluToken] = []
        for k in range(n):
            conllu_tokens.append(
                ConlluToken(
                    id=str(k + 1),
                    form=tokens[k],
                    lemma=(lemmas[k] or "_"),
                    upos=(upos[k] or "_"),
                    xpos="_",
                    feats=(feats[k] or "_"),
                    head=str(int(heads[k])) if heads[k] is not None else "_",
                    deprel=(deprels[k] or "_"),
                    deps="_",
                    misc=misc[k] if k < len(misc) else "_",
                )
            )

        # ✅ Insere linhas MWT antes da primeira palavra de cada grupo.
        # O conll18_ud_eval.py exige a forma de superfície (ex: "pelo") para que
        # a concatenação de tokens do sistema bata com a do gold.
        mwt_info: List[Tuple[int, int, str]] = s.get("mwt") or []
        if mwt_info:
            # Indexado por word_id inicial do MWT → (word_id_final, forma_superficial)
            mwt_by_start: Dict[int, Tuple[int, str]] = {
                start: (end, surface) for start, end, surface in mwt_info
            }
            final_tokens: List[ConlluToken] = []
            for tok in conllu_tokens:
                try:
                    wid = int(tok.id)
                except ValueError:
                    final_tokens.append(tok)
                    continue
                if wid in mwt_by_start:
                    end_id, surface = mwt_by_start[wid]
                    final_tokens.append(
                        ConlluToken(
                            id=f"{wid}-{end_id}",
                            form=surface,
                            lemma="_", upos="_", xpos="_", feats="_",
                            head="_", deprel="_", deps="_", misc="_",
                        )
                    )
                final_tokens.append(tok)
            conllu_tokens = final_tokens

        out.append(ConlluSentence(sent_id=f"sys_{i}", text=sent_text, tokens=conllu_tokens))

    write_conllu(output_path, out)
    logger.info(f"[{framework}] CoNLL-U end-to-end salvo em: {output_path}")


def build_pred_wikiner_bio(
    framework: str,
    model_name: str,
    wikiner_path: str,
    output_path: str,
    max_sentences: int = 5000,
):
    """
    NER em WikiNER avaliado em gold tokens (formato do dataset).
    """
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
    build_pred_conllu_end2end(
        "spacy",
        config["models"]["spacy"],
        gold_path,
        os.path.join(spacy_out, "bosque_pred_e2e.conllu"),
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
    build_pred_conllu_end2end(
        "stanza",
        config["models"]["stanza_lang"],
        gold_path,
        os.path.join(stanza_out, "bosque_pred_e2e.conllu"),
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
    build_pred_conllu_end2end(
        "udpipe",
        config["paths"]["udpipe_model_path"],
        gold_path,
        os.path.join(udpipe_out, "bosque_pred_e2e.conllu"),
    )
    build_pred_wikiner_bio(
        "udpipe",
        config["paths"]["udpipe_model_path"],
        wikiner_path,
        os.path.join(udpipe_out, "wikiner_pred.bio"),
        max_sentences=max_ner,
    )

    logger.info("Todas as pipelines (end-to-end UD) foram executadas com sucesso!")
