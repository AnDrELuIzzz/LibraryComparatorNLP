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

def mcc(counts: Dict[str, int]) -> float:
    """Coeficiente de correlação de Matthews (one-versus-rest)."""
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    return float((TP * TN - FP * FN) / den) if den else 0.0


def im_table_from_aligned(pairs) -> Dict[str, Dict]:
    """BI, MK e MCC por classe UPOS somente sobre pares de palavras alinhados
    pelo conll18_ud_eval.py (ver src/ud_alignment.py).

    Usar só os pares alinhados é necessário porque cada ferramenta tokeniza o
    texto à sua maneira: comparar listas por índice deslocaria todos os rótulos
    após a primeira divergência de tokenização, o que invalidaria BI/MK.
    ``support`` é o número de palavras gold alinhadas de cada classe.
    """
    gold = [p.gold_upos for p in pairs]
    pred = [p.system_upos for p in pairs]
    labels = sorted(set(gold))
    cps = confusion_per_label(gold, pred, sorted(set(gold) | set(pred)))
    out = {}
    for lab in labels:
        c = cps[lab]
        out[lab] = {**c, **informedness_markedness(c), "mcc": mcc(c), "support": int(c["TP"] + c["FN"])}
    return out
