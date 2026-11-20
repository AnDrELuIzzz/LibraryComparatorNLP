# -*- coding: utf-8 -*-
"""Baixa as partições oficiais do UD_Portuguese-Bosque usadas na análise de sobreposição
e (re)gera data/bosque_test_partition_origin.tsv.

Uso:  python scripts/download_ud_bosque.py [--dest data/ud_releases]
"""
import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.bosque_overlap import RELEASES, SPLITS, build_origin_table  # noqa: E402

TEST_IDENTITY_RELEASES = ("2.12", "2.13", "2.14", "2.15", "2.16", "2.17")
URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/r{rel}/pt_bosque-ud-{split}.conllu"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(ROOT, "data", "ud_releases"))
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    for rel in RELEASES:
        for split in SPLITS:
            out = os.path.join(args.dest, f"{split}_r{rel}.conllu")
            if not os.path.exists(out):
                print("baixando", URL.format(rel=rel, split=split))
                urllib.request.urlretrieve(URL.format(rel=rel, split=split), out)
    test = os.path.join(ROOT, "data", "pt_bosque-ud-test.conllu")
    # O teste do estudo deve ser idêntico ao teste das versões 2.12 em diante
    for rel in TEST_IDENTITY_RELEASES:
        out = os.path.join(args.dest, f"test_r{rel}.conllu")
        if not os.path.exists(out):
            urllib.request.urlretrieve(URL.format(rel=rel, split="test"), out)
        print(f"teste do estudo idêntico ao teste r{rel}:", md5(test) == md5(out))
    rows = build_origin_table(test, args.dest)
    print(f"{len(rows)} sentenças gravadas em data/bosque_test_partition_origin.tsv")


if __name__ == "__main__":
    main()
