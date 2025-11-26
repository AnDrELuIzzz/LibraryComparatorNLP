# error_analysis.py

from typing import List, Dict
from collections import defaultdict

def collect_token_level_errors(results: List[Dict]) -> Dict[str, list]:
    pos_errors = []
    lemma_errors = []
    dep_errors = []

    for sent in results:
        sent_id = sent.get("sent_id", "NA")
        text = sent.get("text", "")
        gold_tokens = sent.get("gold_tokens", [])
        gold_pos = sent.get("gold_pos", [])
        gold_lemmas = sent.get("gold_lemmas", [])
        gold_heads = sent.get("gold_heads", [])
        gold_deprels = sent.get("gold_deprels", [])
        pred_pos = sent.get("pred_pos", [])
        pred_lemmas = sent.get("pred_lemmas", [])
        pred_heads = sent.get("pred_heads", [])
        pred_deprels = sent.get("pred_deprels", [])

        length = min(len(gold_tokens), len(pred_pos), len(pred_lemmas), len(pred_heads), len(pred_deprels))
        for i in range(length):
            token = gold_tokens[i]
            if gold_pos[i] != pred_pos[i]:
                pos_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold": gold_pos[i],
                    "predicted": pred_pos[i],
                })
            if gold_lemmas[i] != pred_lemmas[i]:
                lemma_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold": gold_lemmas[i],
                    "predicted": pred_lemmas[i],
                })
            if gold_heads[i] != pred_heads[i] or gold_deprels[i] != pred_deprels[i]:
                dep_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold": (gold_heads[i], gold_deprels[i]),
                    "predicted": (pred_heads[i], pred_deprels[i]),
                })

    return {
        "pos_errors": pos_errors,
        "lemma_errors": lemma_errors,
        "dep_errors": dep_errors,
    }


def save_error_report(errors: Dict[str, list], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ANÁLISE DETALHADA DE ERROS\n\n")
        for name, err_list in errors.items():
            f.write(f"{name} - total {len(err_list)}\n")
            f.write("-" * 60 + "\n")
            for e in err_list[:50]:  # limite para não explodir
                f.write(
                    f"Sentença {e['sent_id']} | pos {e['position']} | token '{e['token']}'\n"
                )
                f.write(f"Texto: {e['text']}\n")
                f.write(f"Gold: {e['gold']} | Pred: {e['predicted']}\n\n")
            f.write("\n")
