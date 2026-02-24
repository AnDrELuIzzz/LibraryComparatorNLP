# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

_METRIC_LINE = re.compile(r"^\s*(Tokens|Sentences|Words|UPOS|Lemmas|UAS|LAS|MLAS)\s*\|", re.IGNORECASE)

def parse_ud_eval_table(stdout_text: str) -> pd.DataFrame:
    """
    Extrai a tabela "Metric | Precision | Recall | F1 Score | AligndAcc" do stdout.
    Não recalcula nada; só parseia texto.
    """
    lines = [ln.rstrip() for ln in stdout_text.splitlines()]
    rows = []
    in_table = False
    for ln in lines:
        if "Metric |" in ln and "F1" in ln:
            in_table = True
            continue
        if in_table:
            if ln.startswith("-----------") or not ln.strip():
                continue
            if "|" not in ln:
                # fim provável
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 5:
                continue
            metric = parts[0]
            if not _METRIC_LINE.match(metric):
                continue
            rows.append({
                "metric": metric,
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1": float(parts[3]),
            })
    return pd.DataFrame(rows)

def main():
    results_path = "results/full_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plots_dir = "results/plots"
    ensure_dir(plots_dir)

    ud = data["ud_eval_official"]
    frames = []
    for sysname, text in ud.items():
        df = parse_ud_eval_table(text)
        if not df.empty:
            df["system"] = sysname
            frames.append(df)
    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not df_all.empty:
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        sub = df_all[df_all["metric"].isin(["UPOS", "Lemmas", "UAS", "LAS", "MLAS"])]
        sns.barplot(data=sub, x="metric", y="f1", hue="system")
        plt.ylim(0, 1.0)
        plt.title("UD Official (conll18_ud_eval.py) - F1 por métrica")
        out = os.path.join(plots_dir, "ud_official_f1.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    # Informedness/Markedness: heatmap (top labels)
    im = data.get("informedness_markedness", {})
    for task, table in im.items():
        df = pd.DataFrame.from_dict(table, orient="index").reset_index().rename(columns={"index": "label"})
        if df.empty:
            continue
        df = df.sort_values("support", ascending=False).head(20)
        mat = df.set_index("label")[["informedness", "markedness"]]
        plt.figure(figsize=(7, 8))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{task} - Informedness/Markedness (Top-20 por suporte)")
        out = os.path.join(plots_dir, f"im_{task}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    print(f"Plots salvos em: {plots_dir}/")

if __name__ == "__main__":
    main()
