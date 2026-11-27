# -*- coding: utf-8 -*-
"""Pipeline completo: executa spaCy, Stanza e UDPipe, avalia e gera a reanálise.

  1. baixa os modelos que faltarem (src/resources.py);
  2. processa o texto bruto do teste do Bosque e o WikiNER (outputs/);
  3. avalia com o conll18_ud_eval.py oficial (results/ud_eval/) e o seqeval (results/ner/);
  4. grava o relatório consolidado (results/full_results.json, results/report.txt);
  5. roda reanalysis.py sobre as saídas recém-geradas (results/reanalysis/).

Uso:  python main.py [--config config.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logger = logging.getLogger("PIPELINE")


def load_config(path: str) -> dict:
    """Lê o config e torna absolutos os caminhos relativos (relativos à raiz do repositório)."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["paths"] = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in config["paths"].items()}
    return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from src.evaluate_ud_official import eval_all_ud
    from src.evaluate_wikiner import eval_all_wikiner
    from src.report import save_report
    from src.resources import ensure_all
    from src.run_pipeline import run_all_pipelines
    from src.version import get_versions
    import reanalysis

    config = load_config(args.config)
    versions = get_versions()
    versions["stanza_models"] = f"recursos {config['models']['stanza_resources_version']}"
    logger.info(f"Versões: {versions}")

    logger.info("1) Conferindo/baixando modelos...")
    ensure_all(config)

    logger.info("2) Rodando pipelines (UD end-to-end + WikiNER)...")
    run_all_pipelines(config)

    logger.info("3) Avaliação UD oficial (conll18_ud_eval.py)...")
    ud_eval_texts = eval_all_ud(config)

    logger.info("4) Avaliação WikiNER (NER)...")
    ner_results = eval_all_wikiner(config)

    logger.info("5) Salvando relatório consolidado...")
    paths = save_report(config, ud_eval_texts, ner_results, versions)
    logger.info(f"Relatórios: {paths}")

    logger.info("6) Reanálise (sobreposição, bootstrap, BI/MK sobre palavras alinhadas)...")
    reanalysis.main(["--outputs-dir", config["paths"]["outputs_dir"],
                     "--results-dir", config["paths"]["results_dir"]], quiet=True)
    logger.info("Concluído.")


if __name__ == "__main__":
    main()
