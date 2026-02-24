# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def confusion_per_label(gold: List[str], pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1

    out = {}
    N = cm.sum()
    for lab in labels:
        i = idx[lab]
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(N - TP - FN - FP)
        out[lab] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    return out

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def informedness_markedness(counts: Dict[str, int]) -> Dict[str, float]:
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    TPR = safe_div(TP, TP + FN)
    TNR = safe_div(TN, TN + FP)
    PPV = safe_div(TP, TP + FP)
    NPV = safe_div(TN, TN + FN)

    informedness = TPR + TNR - 1.0
    markedness = PPV + NPV - 1.0

    return {
        "TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV,
        "informedness": informedness,
        "markedness": markedness
    }

def compute_im_table(gold: List[str], pred: List[str], ignore: set | None = None) -> Dict[str, Dict]:
    ignore = ignore or set()
    labels = sorted(set(gold) | set(pred))
    labels = [l for l in labels if l not in ignore]

    cps = confusion_per_label(gold, pred, labels)
    out = {}
    for lab in labels:
        out[lab] = {**cps[lab], **informedness_markedness(cps[lab])}
    return out
