"""
Avaliação: métricas, K-Fold, testes estatísticos (t-test, bootstrap).
"""

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats


class EvaluationMetrics:
    @staticmethod
    def compute_pos_accuracy(gold: List[str], pred: List[str]) -> float:
        return accuracy_score(gold, pred) if gold else 0.0

    @staticmethod
    def compute_lemma_accuracy(gold: List[str], pred: List[str]) -> float:
        return accuracy_score(gold, pred) if gold else 0.0

    @staticmethod
    def compute_uas(gold_heads: List[int], pred_heads: List[int]) -> float:
        if not gold_heads:
            return 0.0
        correct = sum(1 for g, p in zip(gold_heads, pred_heads) if g == p)
        return correct / len(gold_heads)

    @staticmethod
    def compute_las(
        gold_heads: List[int],
        gold_deprels: List[str],
        pred_heads: List[int],
        pred_deprels: List[str],
    ) -> float:
        if not gold_heads:
            return 0.0
        correct = sum(
            1
            for gh, gr, ph, pr in zip(gold_heads, gold_deprels, pred_heads, pred_deprels)
            if gh == ph and gr == pr
        )
        return correct / len(gold_heads)

    @staticmethod
    def compute_ner_metrics(gold_tags: List[str], pred_tags: List[str]) -> Dict:
        labels = sorted(set(gold_tags + pred_tags) - {"O"})
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_tags, pred_tags, labels=labels, average="weighted", zero_division=0
        )
        return {"precision": precision, "recall": recall, "f1": f1, "support": support}


class StatisticalTests:
    @staticmethod
    def paired_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Dict:
        if len(a) != len(b):
            raise ValueError("Vetores com tamanhos diferentes")
        t, p = stats.ttest_rel(a, b)
        return {
            "test": "paired_ttest",
            "t_statistic": float(t),
            "p_value": float(p),
            "significant": bool(p < alpha),
            "mean_diff": float(np.mean(a - b)),
        }

    @staticmethod
    def bootstrap_diff(
        a: np.ndarray,
        b: np.ndarray,
        rounds: int = 10000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict:
        if len(a) != len(b):
            raise ValueError("Vetores com tamanhos diferentes")
        rng = np.random.default_rng(seed)
        n = len(a)
        diffs = []
        for _ in range(rounds):
            idx = rng.integers(0, n, n)
            diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
        diffs = np.array(diffs)
        obs = float(np.mean(a) - np.mean(b))
        ci_low = float(np.percentile(diffs, 100 * alpha / 2))
        ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
        p_val = 2 * min(
            np.mean(diffs >= 0),
            np.mean(diffs <= 0)
        )
        return {
            "test": "bootstrap",
            "observed_diff": obs,
            "ci": (ci_low, ci_high),
            "p_value": float(p_val),
            "significant": bool(ci_low > 0 or ci_high < 0),
        }


class CrossValidation:
    @staticmethod
    def kfold_split(n_samples: int, n_splits: int = 5, seed: int = 42):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx = np.arange(n_samples)
        return list(kf.split(idx))
