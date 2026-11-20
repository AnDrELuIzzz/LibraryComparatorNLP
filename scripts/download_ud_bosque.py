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
    print("teste do estudo idêntico ao teste r2.12:",
          md5(test) == md5(os.path.join(args.dest, "test_r2.12.conllu")))
    rows = build_origin_table(test, args.dest)
    print(f"{len(rows)} sentenças gravadas em data/bosque_test_partition_origin.tsv")


if __name__ == "__main__":
    main()
