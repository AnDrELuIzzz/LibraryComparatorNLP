# -*- coding: utf-8 -*-
import re

def rulebased_lemmatization(word: str) -> str:
    """
    Lematizador PT-BR simples baseado em regex.
    Ajuste/expanda conforme sua necessidade.
    """
    w = word.lower()

    verb_rules = [
        (r"(ar|er|ir)am$", r"\1"),
        (r"(ar|er|ir)emos$", r"\1"),
        (r"(ar|er|ir)ão$", r"\1"),
        (r"ando$", "ar"),
        (r"endo$", "er"),
        (r"indo$", "ir"),
    ]

    noun_adj_rules = [
        (r"ões$", "ão"),
        (r"s$", ""),
        (r"zinhas?$", ""),
        (r"inhas?$", ""),
    ]

    for pat, repl in verb_rules:
        if re.search(pat, w):
            lemma = re.sub(pat, repl, w)
            if lemma != w:
                return lemma

    for pat, repl in noun_adj_rules:
        if re.search(pat, w):
            lemma = re.sub(pat, repl, w)
            if lemma != w:
                return lemma

    return w
