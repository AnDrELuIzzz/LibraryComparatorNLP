# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple
import logging

from sklearn.metrics import precision_recall_fscore_support

from .data_loader import DataLoader
from .model_wrapper import get_model_wrapper

logger = logging.getLogger(__name__)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def token_level_prf1(gold: List[str], pred: List[str]) -> Dict:
    labels = sorted(set(gold + pred) - {"O"})
    p, r, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "labels": labels,
        "support_total": int(sum(support)) if hasattr(support, "__len__") else None
    }

def iob1_to_iob2(tags: List[str]) -> List[str]:
    """Converte rótulos IOB1 (CoNLL-2003/WikiNER: I- em quase todos os tokens; B- só
    entre entidades adjacentes do mesmo tipo) para IOB2 (B- no primeiro token de toda
    entidade), esquema em que o spaCy produz suas predições (Tjong Kim Sang; Veenstra, 1999)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def read_bio_predictions(path: str) -> List[List[str]]:
    """Lê predições salvas (uma sentença por linha, tokens 'palavra|rótulo')."""
    sents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items = [it for it in line.rstrip("\n").split(" ") if it]
            sents.append([it.rsplit("|", 1)[1] for it in items])
    return sents


def score_ner(gold_sents: List[List[str]], pred_sents: List[List[str]]) -> Dict:
    """Métricas por token (gold em IOB2 e, para transparência, no IOB1 original) e por entidade."""
    gold_flat, pred_flat, g_sents, p_sents = [], [], [], []
    for g, p in zip(gold_sents, pred_sents):
        L = min(len(g), len(p))
        g_sents.append(g[:L]); p_sents.append(p[:L])
        gold_flat.extend(g[:L]); pred_flat.extend(p[:L])
    gold_iob2_sents = [iob1_to_iob2(g) for g in g_sents]
    gold_iob2_flat = [t for g in gold_iob2_sents for t in g]
    return {
        "token_level": token_level_prf1(gold_iob2_flat, pred_flat),
        "token_level_gold_iob1": token_level_prf1(gold_flat, pred_flat),
        "span_level": try_seqeval_spans(gold_iob2_sents, p_sents),
        "span_level_gold_iob1": try_seqeval_spans(g_sents, p_sents),
        "span_level_strict_iob2": try_seqeval_spans(gold_iob2_sents, p_sents, strict=True),
        "n_sentences": len(g_sents),
        "n_tokens": len(gold_flat),
    }


def try_seqeval_spans(gold_sents: List[List[str]], pred_sents: List[List[str]], strict: bool = False) -> Dict | None:
    """Métricas por entidade com o seqeval: modo padrão (compatível com o conlleval)
    ou, com strict=True, modo estrito com o esquema IOB2."""
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
        kw = {}
        if strict:
            from seqeval.scheme import IOB2
            kw = {"mode": "strict", "scheme": IOB2}
        return {
            "precision": float(precision_score(gold_sents, pred_sents, **kw)),
            "recall": float(recall_score(gold_sents, pred_sents, **kw)),
            "f1": float(f1_score(gold_sents, pred_sents, **kw)),
            "report": classification_report(gold_sents, pred_sents, digits=4, **kw),
            "note": "seqeval (span-level, strict IOB2)" if strict else "seqeval (span-level)"
        }
    except Exception:
        return None

def eval_wikiner_one(framework: str, model_name: str, wikiner_path: str, max_sentences: int = 5000) -> Dict:
    loader = DataLoader(conllu_path="(unused)", wikiner_path=wikiner_path)
    samples = loader.load_wikiner(max_sentences=max_sentences)

    wrapper = get_model_wrapper(framework, model_name)
    wrapper.load_model()

    gold_sents = [s["tags"] for s in samples]
    pred_sents = [wrapper.ner_gold(s["tokens"]) for s in samples]

    # O gold do WikiNER está em IOB1; score_ner converte-o para IOB2 antes da
    # avaliação por token (e mantém o resultado com o IOB1 original em
    # "token_level_gold_iob1"). O seqeval por entidade não depende da conversão.
    return {"framework": framework, "model": model_name, **score_ner(gold_sents, pred_sents)}

def eval_all_wikiner(config: Dict) -> Dict:
    results_dir = os.path.join(config["paths"]["results_dir"], "ner")
    ensure_dir(results_dir)

    wikiner_path = config["paths"]["wikiner_path"]
    max_sents = config.get("wikiner", {}).get("max_sentences", 5000)

    systems = {
        "spacy": ("spacy", config["models"]["spacy"]),
        "stanza": ("stanza", config["models"]["stanza_lang"]),
        "udpipe": ("udpipe", config["paths"]["udpipe_model_path"]),
    }

    all_results = {}
    for key, (fw, model) in systems.items():
        metrics = eval_wikiner_one(fw, model, wikiner_path, max_sentences=max_sents)
        all_results[key] = metrics
        out_path = os.path.join(results_dir, f"{key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"[WikiNER] salvo: {out_path}")

    return all_results
