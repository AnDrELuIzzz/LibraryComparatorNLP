# -*- coding: utf-8 -*-
"""Testes de consistência da reanálise contra os resultados oficiais versionados.

Rodar:  python -m pytest -q tests/
Os testes leem apenas arquivos do repositório (saídas salvas em outputs/ e
resultados em results/); nenhum modelo é executado.
"""
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bosque_overlap import load_origin_table, subsets  # noqa: E402
from src.evaluate_wikiner import iob1_to_iob2  # noqa: E402
from src.ud_alignment import OFFICIAL_SCRIPT, OFFICIAL_SCRIPT_MD5  # noqa: E402

SYSTEMS = ("spacy", "stanza", "udpipe")


@pytest.fixture(scope="session")
def reanalysis():
    import reanalysis as ra
    return ra.run_reanalysis(B=2000)  # B menor só para acelerar os testes


def official_table(name):
    """Lê results/ud_eval/<ferramenta>.txt (saída do conll18_ud_eval.py -v)."""
    out = {}
    with open(os.path.join(ROOT, "results", "ud_eval", f"{name}.txt"), encoding="utf-8") as f:
        for line in f:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 4 and cols[1].replace(".", "").isdigit():
                out[cols[0]] = {"precision": float(cols[1]), "recall": float(cols[2]), "f1": float(cols[3])}
    return out


def test_script_oficial_inalterado():
    with open(OFFICIAL_SCRIPT, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == OFFICIAL_SCRIPT_MD5


@pytest.mark.parametrize("name", SYSTEMS)
def test_total_reproduz_recall_oficial(reanalysis, name):
    """Recall do subconjunto 'total' == coluna Recall da Tabela 4 (results/ud_eval)."""
    ref = official_table(name)
    tot = reanalysis["subsets"][name]["total"]
    for metric in ("UPOS", "UFeats", "Lemmas", "UAS", "LAS"):
        assert round(tot[metric]["recall"], 2) == pytest.approx(ref[metric]["recall"], abs=0.005), metric


def test_sobreposicao_bosque():
    origin = load_origin_table()
    s = subsets(origin, "2.5")
    assert len(origin) == 1167
    assert (len(s["treino"]), len(s["desenvolvimento"]), len(s["ineditas"])) == (768, 288, 111)
    assert (len(s["PB"]), len(s["PE"])) == (521, 646)
    assert s["ineditas"] <= s["PB"]          # as inéditas são todas brasileiras
    assert s["PE"] <= s["treino"]            # todas as europeias estavam no treino
    assert all(r["split_2.12"] == "test" and r["split_2.15"] == "test" for r in origin.values())


def test_iob1_para_iob2():
    assert iob1_to_iob2(["I-PER", "I-PER", "O", "I-LOC", "B-LOC", "I-ORG"]) == \
        ["B-PER", "I-PER", "O", "B-LOC", "B-LOC", "B-ORG"]


def test_ner_reproduz_resultado_salvo_e_iob2(reanalysis):
    with open(os.path.join(ROOT, "results", "ner", "spacy.json"), encoding="utf-8") as f:
        saved = json.load(f)
    ner = reanalysis["ner_spacy"]
    for level in ("token_level", "token_level_gold_iob1", "span_level"):
        for k in ("precision", "recall", "f1"):
            assert ner[level][k] == pytest.approx(saved[level][k], abs=1e-9), (level, k)
    assert round(ner["token_level"]["f1"], 4) == 0.9336            # gold IOB2 (Seção 4.3)
    assert round(ner["token_level_gold_iob1"]["recall"], 4) == 0.3760  # gold IOB1 (Seção 4.3)
    assert round(ner["span_level"]["f1"], 4) == 0.9385             # Tabela 5
    for k in ("precision", "recall", "f1"):  # modo estrito IOB2 == modo padrão
        assert ner["span_level_strict_iob2"][k] == pytest.approx(ner["span_level"][k], abs=1e-12)
    assert ner["n_sentences"] == 4701


def test_bootstrap_inversao_nas_ineditas(reanalysis):
    tot = reanalysis["bootstrap"]["udpipe-stanza|total|LAS"]
    ined = reanalysis["bootstrap"]["udpipe-stanza|ineditas|LAS"]
    assert tot["ci95"][0] > 0                 # UDPipe à frente no total
    assert ined["ci95"][1] < 0                # Stanza à frente nas inéditas
    assert round(ined["diff"], 2) == -4.14


def test_bootstrap_deterministico():
    import reanalysis as ra
    from src.paired_bootstrap import paired_bootstrap
    from src.subset_eval import gold_word_sent_ids
    from src.ud_alignment import aligned_words
    ws = gold_word_sent_ids(ra.GOLD)
    a, _, _ = aligned_words(ra.GOLD, ra.system_path("udpipe"))
    b, _, _ = aligned_words(ra.GOLD, ra.system_path("stanza"))
    ids = set(ws)
    r1 = paired_bootstrap(a, b, ws, ids, "LAS", B=500, seed=42)
    r2 = paired_bootstrap(a, b, ws, ids, "LAS", B=500, seed=42)
    assert r1 == r2


def test_bi_mk_alinhados(reanalysis):
    im = reanalysis["im_aligned"]
    assert round(im["spacy"]["ADJ"]["informedness"], 3) == 0.956
    assert round(im["stanza"]["ADP"]["support"]) == 4105
    # com alinhamento, BI de NOUN fica acima de 0,95 nas três ferramentas
    assert all(im[s]["NOUN"]["informedness"] > 0.95 for s in SYSTEMS)


def test_intersecao_e_lemas(reanalysis):
    it = reanalysis["intersection"]
    assert it["common_words"] == 23483
    assert round(it["spacy"]["UPOS"], 2) == 98.13 and round(it["stanza"]["UPOS"], 2) == 97.52
    lc = reanalysis["lemma_case_insensitive"]["spacy"]["case_insensitive"]
    assert round(lc["Lemmas_aligned_acc"], 2) == 98.62


def test_contracoes_nao_expandidas(reanalysis):
    u = reanalysis["unexpanded_contractions"]
    assert u["spacy"]["unexpanded"] == u["spacy"]["gold_multiword_tokens"] == 2015
    assert u["udpipe"]["unexpanded"] == 0


def test_estatisticas_citadas(reanalysis):
    cs = reanalysis["corpus_stats"]
    assert (cs["sentences"], cs["words"], cs["multiword_tokens"]) == (1167, 27604, 2015)
    assert cs["lemmas_with_uppercase"] == 2086
    ner = reanalysis["ner_spacy"]
    assert (ner["n_tokens"], ner["gold_entity_tokens"], ner["gold_B_tags_iob1"]) == (121387, 16030, 13)
    assert reanalysis["subsets"]["spacy"]["ineditas"]["gold_words"] == 2296
    assert reanalysis["subsets"]["spacy"]["fora_do_treino"]["n_sentences"] == 399


def test_avaliacao_ner_do_pipeline_reproduz_json(tmp_path):
    """eval_all_wikiner (usado por main.py) reproduz results/ner/spacy.json a partir das predições salvas."""
    from main import load_config
    from src.evaluate_wikiner import eval_all_wikiner
    config = load_config(os.path.join(ROOT, "config.json"))
    config["paths"]["results_dir"] = str(tmp_path)
    new = eval_all_wikiner(config)["spacy"]
    with open(os.path.join(ROOT, "results", "ner", "spacy.json"), encoding="utf-8") as f:
        saved = json.load(f)
    for level in ("token_level", "token_level_gold_iob1", "span_level", "span_level_strict_iob2"):
        for k in ("precision", "recall", "f1"):
            assert new[level][k] == pytest.approx(saved[level][k], abs=1e-12), (level, k)
