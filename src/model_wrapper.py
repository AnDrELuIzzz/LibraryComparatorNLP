# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import logging

from .lemma_rules import rulebased_lemmatization

logger = logging.getLogger(__name__)

class NLPModelWrapper(ABC):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    # -----------------------------
    # Modo RAW (para CoNLL-U oficial)
    # -----------------------------
    @abstractmethod
    def parse_raw(self, text: str) -> List[Dict]:
        """
        Retorna lista de sentenças:
          [{
            "tokens": [str...],
            "lemmas": [str...],
            "upos": [str...],
            "heads": [int...],     # 1-indexed, 0 = root (por sentença)
            "deprels": [str...],
            "feats": [str...],     # pode ser "_" se não disponível
            "text": str            # texto reconstruído da sentença (opcional)
            "offsets": [(start,end)...] (opcional)
          }, ...]
        """
        raise NotImplementedError

    # -----------------------------
    # Modo GOLD TOKENS (controle)
    # -----------------------------
    @abstractmethod
    def pos_tag_gold(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def lemmatize_gold(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def dependency_parse_gold(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        raise NotImplementedError

    @abstractmethod
    def ner_gold(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError


class SpacyWrapper(NLPModelWrapper):
    def load_model(self):
        import spacy
        from spacy.cli import download
        try:
            self.model = spacy.load(self.model_name)
            if ("parser" not in self.model.pipe_names
                and "senter" not in self.model.pipe_names
                and "sentencizer" not in self.model.pipe_names):
                self.model.add_pipe("sentencizer")
            logger.info(f"spaCy loaded: {self.model_name}")
        except OSError:
            logger.warning(f"Downloading {self.model_name}...")
            download(self.model_name)
            self.model = spacy.load(self.model_name)

    def _ensure(self):
        if self.model is None:
            self.load_model()

    def parse_raw(self, text: str) -> List[Dict]:
        from spacy.tokens import Doc
        self._ensure()
        doc = self.model(text)

        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
        out = []
        for si, sent in enumerate(sents, start=1):
            tokens = [t.text for t in sent]
            lemmas = [(t.lemma_ or t.text).lower() for t in sent]
            upos = [t.pos_ or "_" for t in sent]

            # feats (spaCy morph) -> UD-like "Feat=Val|Feat2=Val2"
            feats = []
            for t in sent:
                if t.morph:
                    items = []
                    md = t.morph.to_dict()
                    for k in sorted(md.keys()):
                        v = md[k]
                        if isinstance(v, list):
                            for vv in v:
                                items.append(f"{k}={vv}")
                        else:
                            items.append(f"{k}={v}")
                    feats.append("|".join(items) if items else "_")
                else:
                    feats.append("_")

            # heads/deprels em índices de sentença (1-indexed), root=0
            heads = []
            deprels = []
            sent_start = sent.start
            for t in sent:
                deprels.append(t.dep_ or "_")
                if t.head.i == t.i:
                    heads.append(0)
                else:
                    heads.append((t.head.i - sent_start) + 1)

            offsets = [(t.idx, t.idx + len(t.text)) for t in sent]
            out.append({
                "tokens": tokens,
                "lemmas": lemmas,
                "upos": upos,
                "heads": heads,
                "deprels": deprels,
                "feats": feats,
                "offsets": offsets,
                "text": sent.text
            })
        return out

    # GOLD mode
    def _process_with_gold_tokens(self, tokens: List[str]):
        from spacy.tokens import Doc
        self._ensure()
        doc = Doc(self.model.vocab, words=tokens)
        for name, proc in self.model.pipeline:
            if name != "tokenizer":
                doc = proc(doc)
        return doc

    def pos_tag_gold(self, tokens: List[str]) -> List[str]:
        doc = self._process_with_gold_tokens(tokens)
        return [t.pos_ for t in doc]

    def lemmatize_gold(self, tokens: List[str]) -> List[str]:
        doc = self._process_with_gold_tokens(tokens)
        lemmas = []
        for tok in doc:
            surface = tok.text
            lemma_spacy = (tok.lemma_ or "").lower()
            if lemma_spacy and lemma_spacy != surface.lower():
                lemmas.append(lemma_spacy)
            else:
                lemmas.append(rulebased_lemmatization(surface))
        return lemmas

    def dependency_parse_gold(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        doc = self._process_with_gold_tokens(tokens)
        heads = [t.head.i + 1 for t in doc]  # doc inteiro (não por sentença)
        deprels = [t.dep_ for t in doc]
        return heads, deprels

    def ner_gold(self, tokens: List[str]) -> List[str]:
        doc = self._process_with_gold_tokens(tokens)
        tags = ["O"] * len(tokens)
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                if i < len(tags):
                    tags[i] = ("B-" if i == ent.start else "I-") + ent.label_
        return tags


class StanzaWrapper(NLPModelWrapper):
    def load_model(self):
        import stanza
        processors = "tokenize,mwt,pos,lemma,depparse"
        try:
            self.model = stanza.Pipeline(self.model_name, processors=processors, download_method=None)
        except Exception:
            stanza.download(self.model_name)
            self.model = stanza.Pipeline(self.model_name, processors=processors)
        logger.info(f"Stanza loaded: {self.model_name} (sem NER)")

    def _ensure(self):
        if self.model is None:
            self.load_model()

    def parse_raw(self, text: str) -> List[Dict]:
        self._ensure()
        doc = self.model(text)
        out = []
        for sent in doc.sentences:
            tokens = [w.text for w in sent.words]
            lemmas = [(w.lemma or w.text).lower() for w in sent.words]
            upos = [w.upos or "_" for w in sent.words]
            heads = [int(w.head) for w in sent.words]  # stanza já usa 0 root, 1..n
            deprels = [w.deprel or "_" for w in sent.words]
            feats = [w.feats if getattr(w, "feats", None) else "_" for w in sent.words]

            # offsets: nem sempre disponíveis em words; se não tiver, deixa None
            offsets = None
            starts = [getattr(w, "start_char", None) for w in sent.words]
            ends = [getattr(w, "end_char", None) for w in sent.words]

            if all(s is not None and e is not None for s, e in zip(starts, ends)):
                offsets = [(int(s), int(e)) for s, e in zip(starts, ends)]
            else:
                offsets = None

            out.append({
                "tokens": tokens,
                "lemmas": lemmas,
                "upos": upos,
                "heads": heads,
                "deprels": deprels,
                "feats": feats,
                "offsets": offsets,
                "text": " ".join(tokens)
            })
        return out

    # GOLD mode (controle; NER inexistente para pt no wrapper)
    def pos_tag_gold(self, tokens: List[str]) -> List[str]:
        self._ensure()
        doc = self.model(" ".join(tokens))
        return [w.upos for sent in doc.sentences for w in sent.words]

    def lemmatize_gold(self, tokens: List[str]) -> List[str]:
        self._ensure()
        doc = self.model(" ".join(tokens))
        return [w.lemma for sent in doc.sentences for w in sent.words]

    def dependency_parse_gold(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        self._ensure()
        doc = self.model(" ".join(tokens))
        heads, deprels = [], []
        for sent in doc.sentences:
            for w in sent.words:
                heads.append(int(w.head))
                deprels.append(w.deprel)
        return heads, deprels

    def ner_gold(self, tokens: List[str]) -> List[str]:
        return ["O"] * len(tokens)


class UDPipeWrapper(NLPModelWrapper):
    def load_model(self):
        import ufal.udpipe as udpipe
        self.udpipe = udpipe
        self.model = udpipe.Model.load(self.model_name)
        if not self.model:
            raise RuntimeError(f"Failed to load UDPipe model: {self.model_name}")
        self.pipeline = udpipe.Pipeline(
            self.model,
            "tokenize",
            udpipe.Pipeline.DEFAULT,
            udpipe.Pipeline.DEFAULT,
            "conllu"
        )
        logger.info(f"UDPipe loaded: {self.model_name}")

    def _ensure(self):
        if self.model is None:
            self.load_model()

    def parse_raw(self, text: str) -> List[Dict]:
        from .conllu_utils import read_conllu, build_document_text_from_gold  # evita circular? (uso leve)
        self._ensure()
        conllu_str = self.pipeline.process(text)

        # parse "string" como se fosse arquivo: solução simples
        sents = []
        cur = []
        meta_sent_id = None
        meta_text = None
        for line in conllu_str.splitlines():
            if not line.strip():
                if cur:
                    sents.append((meta_sent_id, meta_text, cur))
                cur, meta_sent_id, meta_text = [], None, None
                continue
            if line.startswith("# sent_id"):
                meta_sent_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("# text"):
                meta_text = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue
            cur.append(cols)
        if cur:
            sents.append((meta_sent_id, meta_text, cur))

        out = []
        for (sid, stxt, rows) in sents:
            tokens = [r[1] for r in rows]
            lemmas = [(r[2] if r[2] else r[1]).lower() for r in rows]
            upos = [r[3] or "_" for r in rows]
            feats = [r[5] or "_" for r in rows]
            heads = [int(r[6]) if r[6].isdigit() else 0 for r in rows]
            deprels = [r[7] or "_" for r in rows]
            out.append({
                "tokens": tokens,
                "lemmas": lemmas,
                "upos": upos,
                "heads": heads,
                "deprels": deprels,
                "feats": feats,
                "offsets": None,
                "text": stxt or " ".join(tokens)
            })
        return out

    # GOLD mode
    def _process(self, text: str) -> str:
        self._ensure()
        return self.pipeline.process(text)

    def _field(self, conllu: str, idx: int, as_int: bool = False):
        vals = []
        for line in conllu.strip().split("\n"):
            if line and not line.startswith("#") and line[0].isdigit():
                parts = line.split("\t")
                if "-" in parts[0] or "." in parts[0]:
                    continue
                val = parts[idx] if idx < len(parts) else "_"
                if as_int:
                    try:
                        val = int(val)
                    except Exception:
                        val = 0
                vals.append(val)
        return vals

    def pos_tag_gold(self, tokens: List[str]) -> List[str]:
        conllu = self._process(" ".join(tokens))
        return self._field(conllu, 3)

    def lemmatize_gold(self, tokens: List[str]) -> List[str]:
        conllu = self._process(" ".join(tokens))
        return self._field(conllu, 2)

    def dependency_parse_gold(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        conllu = self._process(" ".join(tokens))
        heads = self._field(conllu, 6, as_int=True)
        deprels = self._field(conllu, 7)
        return heads, deprels

    def ner_gold(self, tokens: List[str]) -> List[str]:
        return ["O"] * len(tokens)


def get_model_wrapper(framework: str, model_name: str, device: str = "cpu") -> NLPModelWrapper:
    f = framework.lower()
    if f == "spacy":
        return SpacyWrapper(model_name, device)
    if f == "stanza":
        return StanzaWrapper(model_name, device)
    if f == "udpipe":
        return UDPipeWrapper(model_name, device)
    raise ValueError(f"Unknown framework: {framework}")
