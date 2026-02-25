# -*- coding: utf-8 -*-
import json
import logging
import os

from src.version import print_versions
from src.run_pipeline import run_all_pipelines
from src.evaluate_ud_official import eval_all_ud
from src.evaluate_wikiner import eval_all_wikiner
from src.informedness_markedness import compute_im_table
from src.conllu_utils import read_conllu
from src.model_wrapper import get_model_wrapper
from src.report import save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PIPELINE")


def controlled_im_analysis(config: dict) -> dict:
    """
    Mantém sua análise controlada (gold tokens) para IM em UPOS e NER.
    """
    ud_gold = config["paths"]["ud_gold_conllu"]
    gold_sents = read_conllu(ud_gold)

    gold_upos = []
    tokens_flat = []
    for s in gold_sents:
        for t in s.tokens:
            gold_upos.append(t.upos)
            tokens_flat.append(t.form)

    systems = {
        "spacy": ("spacy", config["models"]["spacy"]),
        "stanza": ("stanza", config["models"]["stanza_lang"]),
        "udpipe": ("udpipe", config["paths"]["udpipe_model_path"]),
    }

    out = {}

    # UPOS: gold tokens
    for key, (fw, model) in systems.items():
        wrapper = get_model_wrapper(fw, model)
        wrapper.load_model()
        pred_upos = wrapper.pos_tag_gold(tokens_flat)
        L = min(len(gold_upos), len(pred_upos))
        table = compute_im_table(gold_upos[:L], pred_upos[:L], ignore={"_"})
        for lab, d in table.items():
            d["support"] = int(d["TP"] + d["FN"])
        out[f"upos_{key}"] = table

    # NER: WikiNER gold tokens
    from src.data_loader import DataLoader
    loader = DataLoader(conllu_path=ud_gold, wikiner_path=config["paths"]["wikiner_path"])
    samples = loader.load_wikiner(max_sentences=config.get("wikiner", {}).get("max_sentences", 5000))

    for key, (fw, model) in systems.items():
        wrapper = get_model_wrapper(fw, model)
        wrapper.load_model()

        gold_tags = []
        pred_tags = []
        for s in samples:
            tokens = s["tokens"]
            g = s["tags"]
            p = wrapper.ner_gold(tokens)
            L = min(len(g), len(p))
            gold_tags.extend(g[:L])
            pred_tags.extend(p[:L])

        table = compute_im_table(gold_tags, pred_tags, ignore={"O"})
        for lab, d in table.items():
            d["support"] = int(d["TP"] + d["FN"])
        out[f"ner_{key}"] = table

    return out


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    versions = print_versions()

    logger.info("1) Rodando pipelines (UD end-to-end + WikiNER)...")
    run_all_pipelines(config)

    logger.info("2) Avaliação UD oficial (conll18_ud_eval.py)...")
    ud_eval_texts = eval_all_ud(config)

    logger.info("3) Avaliação WikiNER (NER)...")
    ner_results = eval_all_wikiner(config)

    logger.info("4) Informedness/Markedness (análise controlada)...")
    im_results = controlled_im_analysis(config)

    logger.info("5) Salvando relatório consolidado...")
    paths = save_report(config, ud_eval_texts, ner_results, im_results, versions)
    logger.info(f"Relatórios: {paths}")

    logger.info("6) Plotando gráficos...")
    os.system("python -m src.plot_results")


if __name__ == "__main__":
    main()
