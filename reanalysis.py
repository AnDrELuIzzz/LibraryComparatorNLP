# -*- coding: utf-8 -*-
"""Reanálise a partir das saídas salvas (não reexecuta nenhum modelo).

Gera, em results/reanalysis/, os números das análises complementares do artigo:
  - sobreposição entre o teste do Bosque e as partições de treino das versões 2.5/2.8;
  - recall e AlignedAcc por subconjunto de sentenças (vistas no treino x inéditas);
  - bootstrap pareado por sentenças (B = 10.000, semente 42) das diferenças entre ferramentas;
  - acurácia sobre as palavras alinhadas pelas três ferramentas;
  - Lemmas/BLEX sem distinção entre maiúsculas e minúsculas;
  - BI, MK e MCC por classe UPOS sobre palavras alinhadas;
  - NER do spaCy por token (gold IOB1 original e convertido para IOB2) e por entidade.

Uso:  python reanalysis.py [--outputs-dir outputs] [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bosque_overlap import load_origin_table, overlap_summary, subsets
from src.data_loader import load_wikiner
from src.evaluate_wikiner import read_bio_predictions, score_ner
from src.informedness_markedness import im_table_from_aligned
from src.paired_bootstrap import DEFAULT_B, DEFAULT_SEED, paired_bootstrap
from src.subset_eval import (gold_word_sent_ids, intersection_accuracy, lemma_case_insensitive,
                             mwt_count, subset_scores, unexpanded_contractions)
from src.ud_alignment import aligned_words

ROOT = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ("spacy", "stanza", "udpipe")
GOLD = os.path.join(ROOT, "data", "pt_bosque-ud-test.conllu")
OUTPUTS = os.path.join(ROOT, "outputs")
RESULTS = os.path.join(ROOT, "results")
WIKINER = os.path.join(ROOT, "data", "aij-wikiner-pt-wp3")
WIKINER_MAX_LINES = 5000  # mesmo valor de config["wikiner"]["max_sentences"] usado no pipeline


def system_path(name: str, outputs_dir: str = OUTPUTS) -> str:
    return os.path.join(outputs_dir, name, "bosque_pred_e2e.conllu")


def bosque_stats(gold_path: str, word_sent) -> Dict:
    """Estatísticas descritivas do teste do Bosque citadas no artigo."""
    n_mwt, upper_lemmas = 0, 0
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 10:
                continue
            if "-" in cols[0]:
                n_mwt += 1
            elif cols[0].isdigit() and cols[2] != cols[2].lower():
                upper_lemmas += 1
    return {"sentences": len(set(word_sent)), "words": len(word_sent), "multiword_tokens": n_mwt,
            "lemmas_with_uppercase": upper_lemmas, "lemmas_with_uppercase_pct": 100 * upper_lemmas / len(word_sent)}


def run_reanalysis(B: int = DEFAULT_B, seed: int = DEFAULT_SEED, outputs_dir: str = OUTPUTS) -> Dict:
    word_sent = gold_word_sent_ids(GOLD)
    origin = load_origin_table()
    subs = subsets(origin, "2.5")

    pairs = {}
    official = {}
    for name in SYSTEMS:
        pairs[name], _, scores = aligned_words(GOLD, system_path(name, outputs_dir))
        official[name] = {k: {"precision": 100 * v.precision, "recall": 100 * v.recall, "f1": 100 * v.f1}
                          for k, v in scores.items()}

    out: Dict = {"overlap": overlap_summary(origin), "official": official}

    out["subsets"] = {name: {s: subset_scores(pairs[name], word_sent, ids) for s, ids in subs.items()}
                      for name in SYSTEMS}

    comparisons = {}
    for a, b in (("udpipe", "stanza"), ("stanza", "spacy"), ("udpipe", "spacy")):
        for s in ("total", "ineditas", "fora_do_treino", "treino"):
            for metric in ("UPOS", "UAS", "LAS"):
                comparisons[f"{a}-{b}|{s}|{metric}"] = paired_bootstrap(
                    pairs[a], pairs[b], word_sent, subs[s], metric, B=B, seed=seed)
    out["bootstrap"] = comparisons

    out["intersection"] = intersection_accuracy(pairs)
    out["lemma_case_insensitive"] = {n: lemma_case_insensitive(GOLD, system_path(n, outputs_dir)) for n in SYSTEMS}
    out["mwt_lines"] = {n: mwt_count(system_path(n, outputs_dir)) for n in SYSTEMS}
    out["unexpanded_contractions"] = {n: unexpanded_contractions(GOLD, system_path(n, outputs_dir)) for n in SYSTEMS}
    out["im_aligned"] = {n: im_table_from_aligned(pairs[n]) for n in SYSTEMS}

    out["corpus_stats"] = bosque_stats(GOLD, word_sent)

    gold_ner = [s["tags"] for s in load_wikiner(WIKINER, max_sentences=WIKINER_MAX_LINES)]
    pred_ner = read_bio_predictions(os.path.join(outputs_dir, "spacy", "wikiner_pred.bio"))
    if len(gold_ner) != len(pred_ner):
        raise ValueError("Predições do WikiNER desalinhadas com o gold")
    ner = score_ner(gold_ner, pred_ner)
    flat = [t for sent in gold_ner for t in sent]
    ner["gold_entity_tokens"] = sum(t != "O" for t in flat)
    ner["gold_B_tags_iob1"] = sum(t.startswith("B-") for t in flat)
    for k in ("span_level", "span_level_gold_iob1", "span_level_strict_iob2"):
        if ner.get(k):
            ner[k].pop("report", None)
    out["ner_spacy"] = ner
    return out


def fmt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def to_markdown(r: Dict) -> str:
    cs = r["corpus_stats"]
    L = ["# Reanálise (gerado por reanalysis.py)", "",
         f"Teste do Bosque: {cs['sentences']} sentenças, {cs['words']} palavras sintáticas, "
         f"{cs['multiword_tokens']} tokens multipalavra; lemas com maiúscula: {cs['lemmas_with_uppercase']} "
         f"({fmt(cs['lemmas_with_uppercase_pct'])}% das palavras).", ""]
    ov = r["overlap"]["by_release"]
    L += ["## Sobreposição do teste (UD 2.12) com as partições anteriores", "",
          "| Versão | treino | desenvolvimento | teste | ausente |", "|---|---|---|---|---|"]
    for rel, d in ov.items():
        cell = lambda k: f"{d[k]['n']} ({fmt(d[k]['pct'])}%)" if k in d else "0"
        L.append(f"| {rel} | {cell('train')} | {cell('dev')} | {cell('test')} | {cell('absent')} |")
    L += ["", f"Variedade × partição 2.5: {r['overlap']['variety_by_split_2.5']}", ""]

    L += ["## Recall (%) por subconjunto (Tabela 6 do artigo)", "",
          "| Subconjunto (sentenças) | spaCy UPOS / UAS / LAS | Stanza UPOS / UAS / LAS | UDPipe UPOS / UAS / LAS |",
          "|---|---|---|---|"]
    labels = {"treino": "Treino", "desenvolvimento": "Desenvolvimento", "ineditas": "Teste, inéditas", "total": "Total"}
    for s, lab in labels.items():
        n = r["subsets"]["spacy"][s]["n_sentences"]
        cells = [" / ".join(fmt(r["subsets"][sy][s][m]["recall"]) for m in ("UPOS", "UAS", "LAS")) for sy in SYSTEMS]
        L.append(f"| {lab} ({n}) | " + " | ".join(cells) + " |")

    L += ["", "## Bootstrap pareado (diferença em pontos; IC95% por percentis)", "",
          "| Comparação | Subconjunto | Métrica | A | B | A − B | IC95% |", "|---|---|---|---|---|---|---|"]
    for key, v in r["bootstrap"].items():
        comp, s, m = key.split("|")
        L.append(f"| {comp} | {s} | {m} | {fmt(v['a'])} | {fmt(v['b'])} | {v['diff']:+.2f} | "
                 f"[{v['ci95'][0]:+.2f}; {v['ci95'][1]:+.2f}] |".replace(".", ","))

    it = r["intersection"]
    L += ["", f"## Acurácia sobre as {it['common_words']} palavras alinhadas pelas três ferramentas", "",
          "| Ferramenta | UPOS | UFeats | Lemas (sem caixa) | UAS | LAS |", "|---|---|---|---|---|---|"]
    for sy in SYSTEMS:
        d = it[sy]
        L.append(f"| {sy} | {fmt(d['UPOS'])} | {fmt(d['UFeats'])} | {fmt(d['Lemmas_ci'])} | {fmt(d['UAS'])} | {fmt(d['LAS'])} |")

    L += ["", "## Lemas sem distinção entre maiúsculas e minúsculas", "",
          "| Ferramenta | Lemmas F1 | Lemmas AlignedAcc | BLEX F1 |", "|---|---|---|---|"]
    for sy in SYSTEMS:
        o, c = r["lemma_case_insensitive"][sy]["original"], r["lemma_case_insensitive"][sy]["case_insensitive"]
        L.append(f"| {sy} | {fmt(o['Lemmas_f1'])} → {fmt(c['Lemmas_f1'])} | "
                 f"{fmt(o['Lemmas_aligned_acc'])} → {fmt(c['Lemmas_aligned_acc'])} | {fmt(o['BLEX_f1'])} → {fmt(c['BLEX_f1'])} |")
    L += ["", f"Linhas de tokens multipalavra nos arquivos preditos: {r['mwt_lines']}",
          f"Tokens multipalavra da referência deixados sem expansão: {r['unexpanded_contractions']}", ""]

    L += ["## BI e MK por classe UPOS sobre palavras alinhadas (Seção 4.2 do artigo)", "",
          "| Classe | spaCy BI | spaCy MK | Stanza BI | Stanza MK | UDPipe BI | UDPipe MK | n (Sp / St / UD) |",
          "|---|---|---|---|---|---|---|---|"]
    classes = sorted(set().union(*(r["im_aligned"][sy].keys() for sy in SYSTEMS)))
    for c in classes:
        row = []
        for sy in SYSTEMS:
            d = r["im_aligned"][sy].get(c)
            row += [f"{round(d['informedness'], 3) + 0.0:.3f}", f"{round(d['markedness'], 3) + 0.0:.3f}"] if d else ["–", "–"]
        sup = " / ".join(str(r["im_aligned"][sy].get(c, {}).get("support", 0)) for sy in SYSTEMS)
        L.append(f"| {c} | " + " | ".join(x.replace(".", ",") for x in row) + f" | {sup} |")

    ner = r["ner_spacy"]
    L += ["", "## NER (spaCy, WikiNER)", "",
          f"- Sentenças: {ner['n_sentences']}; tokens: {ner['n_tokens']}; tokens de entidade no gold: "
          f"{ner['gold_entity_tokens']}; rótulos B- no gold IOB1: {ner['gold_B_tags_iob1']}",
          "- Por token, gold IOB1 original (média ponderada, sem O): P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**ner["token_level_gold_iob1"]),
          "- Por token, gold convertido para IOB2: P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**ner["token_level"]),
          "- Por entidade (seqeval, modo padrão): P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**ner["span_level"]),
          "- Por entidade (seqeval, modo estrito IOB2): P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**ner["span_level_strict_iob2"]), ""]
    return "\n".join(L)


def main(argv=None, quiet: bool = False):
    ap = argparse.ArgumentParser(description="Reanálise a partir das saídas salvas (sem rodar modelos).")
    ap.add_argument("--outputs-dir", default=OUTPUTS, help="onde estão <ferramenta>/bosque_pred_e2e.conllu")
    ap.add_argument("--results-dir", default=RESULTS, help="a reanálise é gravada em <results-dir>/reanalysis/")
    args = ap.parse_args(argv)
    r = run_reanalysis(outputs_dir=args.outputs_dir)
    out_dir = os.path.join(args.results_dir, "reanalysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reanalysis.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    md = to_markdown(r)
    with open(os.path.join(out_dir, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if not quiet:
        print(md)
    print(f"Reanálise gravada em {out_dir}/")


if __name__ == "__main__":
    main()
