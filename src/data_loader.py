"""
Data Loader: carrega CONLL-U (UD Portuguese Bosque) e WikiNER.
"""

from typing import List, Dict


class DataLoader:
    def __init__(self, conllu_path: str, wikiner_path: str | None = None):
        self.conllu_path = conllu_path
        self.wikiner_path = wikiner_path

    def load_conllu(self) -> List[Dict]:
        """
        Retorna lista de dicts:
        {
          "text": str,
          "tokens": [...],
          "pos": [...],
          "lemmas": [...],
          "heads": [...],
          "deprels": [...]
        }
        """
        sentences = []
        current = {
            "text": "",
            "tokens": [],
            "pos": [],
            "lemmas": [],
            "heads": [],
            "deprels": [],
        }

        with open(self.conllu_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# text ="):
                    current["text"] = line.split("=", 1)[1].strip()
                elif line.startswith("# sent_id"):
                    # apenas armazenar se já tiver tokens
                    if current["tokens"]:
                        sentences.append(current)
                        current = {
                            "text": "",
                            "tokens": [],
                            "pos": [],
                            "lemmas": [],
                            "heads": [],
                            "deprels": [],
                        }
                elif line and line[0].isdigit() and "-" not in line.split("\t")[0]:
                    parts = line.split("\t")
                    token = parts[1]
                    lemma = parts[2]
                    upos = parts[3]
                    head = int(parts[6])
                    deprel = parts[7]
                    current["tokens"].append(token)
                    current["lemmas"].append(lemma)
                    current["pos"].append(upos)
                    current["heads"].append(head)
                    current["deprels"].append(deprel)
                elif not line:
                    if current["tokens"]:
                        sentences.append(current)
                        current = {
                            "text": "",
                            "tokens": [],
                            "pos": [],
                            "lemmas": [],
                            "heads": [],
                            "deprels": [],
                        }
        if current["tokens"]:
            sentences.append(current)
        return sentences

    def load_wikiner(self, max_sentences: int | None = None) -> List[Dict]:
        """
        Lê WikiNER formato: palavra|pos|ner por token (linha = sentença).
        Retorna:
        { "tokens": [...], "tags": [...] }
        """
        if not self.wikiner_path:
            return []
        sentences = []
        with open(self.wikiner_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_sentences and i >= max_sentences:
                    break
                parts = [p for p in line.strip().split(" ") if p]
                if not parts:
                    continue
                tokens, tags = [], []
                for tok in parts:
                    try:
                        word, pos, ner = tok.split("|")
                    except ValueError:
                        continue
                    tokens.append(word)
                    tags.append(ner)
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags})
        return sentences
