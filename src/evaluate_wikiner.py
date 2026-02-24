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

def try_seqeval_spans(gold_sents: List[List[str]], pred_sents: List[List[str]]) -> Dict | None:
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
        return {
            "precision": float(precision_score(gold_sents, pred_sents)),
            "recall": float(recall_score(gold_sents, pred_sents)),
            "f1": float(f1_score(gold_sents, pred_sents)),
            "report": classification_report(gold_sents, pred_sents, digits=4),
            "note": "seqeval (span-level)"
        }
    except Exception:
        return None

def eval_wikiner_one(framework: str, model_name: str, wikiner_path: str, max_sentences: int = 5000) -> Dict:
    loader = DataLoader(conllu_path="(unused)", wikiner_path=wikiner_path)
    samples = loader.load_wikiner(max_sentences=max_sentences)

    wrapper = get_model_wrapper(framework, model_name)
    wrapper.load_model()

    gold_flat, pred_flat = [], []
    gold_sents, pred_sents = [], []

    for s in samples:
        tokens = s["tokens"]
        gold = s["tags"]
        pred = wrapper.ner_gold(tokens)
        L = min(len(gold), len(pred))
        gold = gold[:L]
        pred = pred[:L]

        gold_flat.extend(gold)
        pred_flat.extend(pred)
        gold_sents.append(gold)
        pred_sents.append(pred)

    span_metrics = try_seqeval_spans(gold_sents, pred_sents)
    tok_metrics = token_level_prf1(gold_flat, pred_flat)

    return {
        "framework": framework,
        "model": model_name,
        "token_level": tok_metrics,
        "span_level": span_metrics
    }

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
