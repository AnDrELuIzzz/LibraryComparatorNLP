# -*- coding: utf-8 -*-
"""Adaptadores das três ferramentas para uma interface comum.

``parse_raw(texto)`` devolve, por sentença, um dicionário com tokens, lemas (em minúsculas),
UPOS, traços, núcleos, relações, offsets e tokens multipalavra (MWT). Só o spaCy tem NER
para o português; ``SpacyWrapper.ner_gold`` rotula tokens já segmentados (WikiNER).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class NLPModelWrapper(ABC):
    def __init__(self, model_name: str, model_dir: str | None = None):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = None

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def parse_raw(self, text: str) -> List[Dict]:
        raise NotImplementedError


class SpacyWrapper(NLPModelWrapper):
    def load_model(self):
        import spacy
        from spacy.cli import download

        try:
            self.model = spacy.load(self.model_name)
            if (
                "parser" not in self.model.pipe_names
                and "senter" not in self.model.pipe_names
                and "sentencizer" not in self.model.pipe_names
            ):
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
        self._ensure()
        doc = self.model(text)

        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
        out = []
        for sent in sents:
            # ✅ Filtra tokens puramente whitespace gerados pelo \n separador de sentenças
            real_toks = [t for t in sent if t.text.strip()]
            if not real_toks:
                continue

            # Mapeamento: índice absoluto no doc → posição 1-based na lista filtrada
            tok_idx_map: Dict[int, int] = {
                t.i: new_i for new_i, t in enumerate(real_toks, start=1)
            }

            tokens = [t.text for t in real_toks]
            lemmas = [(t.lemma_ or t.text).lower() for t in real_toks]
            upos = [t.pos_ or "_" for t in real_toks]

            feats = []
            for t in real_toks:
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

            heads = []
            deprels = []
            for t in real_toks:
                deprels.append(t.dep_ or "_")
                if t.head.i == t.i:
                    heads.append(0)
                else:
                    heads.append(tok_idx_map.get(t.head.i, 0))

            offsets = [(t.idx, t.idx + len(t.text)) for t in real_toks]

            # spaCy não produz MWT em português — lista vazia
            out.append(
                {
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "upos": upos,
                    "heads": heads,
                    "deprels": deprels,
                    "feats": feats,
                    "offsets": offsets,
                    "text": sent.text.strip(),
                    "sent_start": int(real_toks[0].idx),
                    "sent_end": int(real_toks[-1].idx + len(real_toks[-1].text)),
                    "mwt": [],
                }
            )
        return out

    def _process_with_gold_tokens(self, tokens: List[str]):
        from spacy.tokens import Doc

        self._ensure()
        doc = Doc(self.model.vocab, words=tokens)
        for name, proc in self.model.pipeline:
            if name != "tokenizer":
                doc = proc(doc)
        return doc

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

        # download_method=None: usa só os modelos já presentes em model_dir (baixados por
        # src/resources.py na versão fixada em config.json), sem atualizá-los.
        kw = {"dir": self.model_dir} if self.model_dir else {}
        self.model = stanza.Pipeline(self.model_name, processors="tokenize,mwt,pos,lemma,depparse",
                                     download_method=None, **kw)
        logger.info(f"Stanza loaded: {self.model_name} ({self.model_dir or 'diretório padrão'}; sem NER)")

    def _ensure(self):
        if self.model is None:
            self.load_model()

    def parse_raw(self, text: str) -> List[Dict]:
        self._ensure()
        doc = self.model(text)
        out = []

        for sent in doc.sentences:
            # ✅ Captura informação de MWT a partir de sent.tokens (forma superficial).
            # Ex: "pelo" é 1 token de superfície que expande para 2 palavras ("por", "o").
            # O conll18_ud_eval.py concatena as formas de superfície (MWT) para verificar
            # a cobertura, então precisamos emitir a linha "1-2  pelo  ..." no CoNLL-U.
            mwt: List[Tuple[int, int, str]] = []
            word_offset = 0
            for token in sent.tokens:
                n_words = len(token.words)
                if n_words > 1:
                    start_1based = word_offset + 1
                    end_1based = word_offset + n_words
                    mwt.append((start_1based, end_1based, token.text))
                word_offset += n_words

            tokens = [w.text for w in sent.words]
            lemmas = [(w.lemma or w.text).lower() for w in sent.words]
            upos = [w.upos or "_" for w in sent.words]
            heads = [int(w.head) for w in sent.words]
            deprels = [w.deprel or "_" for w in sent.words]
            feats = [w.feats if getattr(w, "feats", None) else "_" for w in sent.words]

            starts = [getattr(w, "start_char", None) for w in sent.words]
            ends = [getattr(w, "end_char", None) for w in sent.words]
            if all(s is not None and e is not None for s, e in zip(starts, ends)):
                offsets = [(int(s), int(e)) for s, e in zip(starts, ends)]
                sent_start = min(int(s) for s in starts)
                sent_end = max(int(e) for e in ends)
            else:
                offsets = None
                sent_start = None
                sent_end = None

            out.append(
                {
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "upos": upos,
                    "heads": heads,
                    "deprels": deprels,
                    "feats": feats,
                    "offsets": offsets,
                    "text": " ".join(tokens),
                    "sent_start": sent_start,
                    "sent_end": sent_end,
                    "mwt": mwt,  # ✅ lista de (start_1based, end_1based, surface_form)
                }
            )
        return out


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
            "conllu",
        )
        logger.info(f"UDPipe loaded: {self.model_name}")

    def _ensure(self):
        if self.model is None:
            self.load_model()

    def parse_raw(self, text: str) -> List[Dict]:
        self._ensure()
        conllu_str = self.pipeline.process(text)

        sents = []
        cur_words = []
        cur_mwt: List[Tuple[int, int, str]] = []
        meta_text = None

        for line in conllu_str.splitlines():
            if not line.strip():
                if cur_words:
                    sents.append((meta_text, cur_words, cur_mwt))
                cur_words, cur_mwt, meta_text = [], [], None
                continue
            if line.startswith("# text"):
                meta_text = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tok_id = cols[0]
            if "." in tok_id:
                continue
            # ✅ Captura linhas MWT em vez de descartá-las
            if "-" in tok_id:
                parts = tok_id.split("-")
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    cur_mwt.append((start, end, cols[1]))
                except ValueError:
                    pass
                continue
            cur_words.append(cols)

        if cur_words:
            sents.append((meta_text, cur_words, cur_mwt))

        out = []
        for (stxt, rows, mwt_list) in sents:
            tokens = [r[1] for r in rows]
            lemmas = [((r[2] if r[2] else r[1]).lower()) for r in rows]
            upos = [r[3] or "_" for r in rows]
            feats = [r[5] or "_" for r in rows]
            heads = [int(r[6]) if r[6].isdigit() else 0 for r in rows]
            deprels = [r[7] or "_" for r in rows]

            out.append(
                {
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "upos": upos,
                    "heads": heads,
                    "deprels": deprels,
                    "feats": feats,
                    "offsets": None,
                    "text": stxt or " ".join(tokens),
                    "sent_start": None,
                    "sent_end": None,
                    "mwt": mwt_list,  # ✅ lista de (start_1based, end_1based, surface_form)
                }
            )
        return out


def get_model_wrapper(framework: str, model_name: str, model_dir: str | None = None) -> NLPModelWrapper:
    f = framework.lower()
    if f == "spacy":
        return SpacyWrapper(model_name)
    if f == "stanza":
        return StanzaWrapper(model_name, model_dir)
    if f == "udpipe":
        return UDPipeWrapper(model_name)
    raise ValueError(f"Unknown framework: {framework}")
