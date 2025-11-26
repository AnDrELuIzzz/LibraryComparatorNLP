# lemma_rules.py

import re

def rulebased_lemmatization(word: str) -> str:
    """
    Versão compacta baseada nas suas regras de comparador_conllu_hugging.py.
    Adapte substituindo pelas suas regras completas de verbos/substantivos/adjetivos.
    """
    w = word.lower()

    # Exemplos básicos (substitua pelas suas listas completas):
    verb_rules = [
        (r"(ar|er|ir)am$", r"\1"),  # cantaram -> cantar
        (r"(ar|er|ir)emos$", r"\1"),  # cantaremos -> cantar
        (r"(ar|er|ir)ão$", r"\1"),  # cantarão -> cantar
        (r"ando$", "ar"),  # cantando -> cantar
        (r"endo$", "er"),  # comendo -> comer
        (r"indo$", "ir"),  # partindo -> partir
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
