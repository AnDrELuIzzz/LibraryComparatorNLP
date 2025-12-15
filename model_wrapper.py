from abc import ABC, abstractmethod
from typing import List, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# Importar suas regras de lematização
from lemma_rules import rulebased_lemmatization


class NLPModelWrapper(ABC):
    """Interface abstrata para modelos de PLN."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def pos_tag(self, tokens: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def lemmatize(self, tokens: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def dependency_parse(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        pass
    
    @abstractmethod
    def ner(self, tokens: List[str]) -> List[str]:
        pass


# ==================== SPACY WRAPPER ====================

class SpacyWrapper(NLPModelWrapper):
    """Wrapper para spaCy com suporte a tokens gold (pré-tokenizados)."""
    
    def load_model(self):
        """Carrega modelo spaCy."""
        import spacy
        from spacy.cli import download
        
        try:
            self.model = spacy.load(self.model_name)
            logger.info(f"spaCy loaded: {self.model_name}")
        except OSError:
            logger.warning(f"Downloading {self.model_name}...")
            download(self.model_name)
            self.model = spacy.load(self.model_name)
    
    def _ensure_model(self):
        if self.model is None:
            self.load_model()
    
    def _process_with_gold_tokens(self, tokens: List[str]):
        """
        Cria um Doc do spaCy a partir de tokens pré-definidos,
        SEM retokenizar, e processa o pipeline.
        """
        from spacy.tokens import Doc
        
        self._ensure_model()
        
        # Cria Doc com tokens gold
        doc = Doc(self.model.vocab, words=tokens)
        
        # Processa pipeline (tagger, parser, etc.) SEM tokenizer
        for name, proc in self.model.pipeline:
            if name != 'tokenizer':  # Pula tokenizer
                doc = proc(doc)
        
        return doc
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenização padrão spaCy (para quando não há tokens gold)."""
        self._ensure_model()
        doc = self.model(text)
        return [t.text for t in doc]
    
    def pos_tag(self, tokens: List[str]) -> List[str]:
        """POS tagging usando tokens gold."""
        doc = self._process_with_gold_tokens(tokens)
        return [t.pos_ for t in doc]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lematização usando tokens gold."""
        doc = self._process_with_gold_tokens(tokens)
        lemmas: List[str] = []
        
        for tok in doc:
            surface = tok.text
            lemma_spacy = tok.lemma_.lower()
            surface_low = surface.lower()
            
            # Se spaCy retorna algo diferente, usa
            if lemma_spacy and lemma_spacy != surface_low:
                lemmas.append(lemma_spacy)
            else:
                # Fallback: regras PT-BR
                lemmas.append(rulebased_lemmatization(surface))
        
        return lemmas
    
    def dependency_parse(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        """Parsing de dependências usando tokens gold."""
        doc = self._process_with_gold_tokens(tokens)
        heads = [t.head.i + 1 for t in doc]  # 1-indexed
        deprels = [t.dep_ for t in doc]
        return heads, deprels
    
    def ner(self, tokens: List[str]) -> List[str]:
        """
        NER usando tokens gold.
        Retorna tags no formato IOB2 (B-TIPO, I-TIPO, O).
        """
        doc = self._process_with_gold_tokens(tokens)
        tags = ["O"] * len(tokens)
        
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                if i < len(tags):
                    prefix = "B-" if i == ent.start else "I-"
                    tags[i] = prefix + ent.label_
        
        return tags


# ==================== STANZA WRAPPER ====================

class StanzaWrapper(NLPModelWrapper):
    """Wrapper para Stanza (Português não possui modelo NER oficial)."""

    def load_model(self):
        import stanza
        
        try:
            # Removemos 'ner' porque não há modelo pt disponível
            processors = "tokenize,mwt,pos,lemma,depparse"
            try:
                self.model = stanza.Pipeline(
                    self.model_name,
                    processors=processors,
                    download_method=None,
                )
                logger.info(f"Stanza loaded (sem NER): {self.model_name}")
            except Exception as e:
                logger.warning(f"Error: {e}. Trying download...")
                stanza.download(self.model_name)
                self.model = stanza.Pipeline(
                    self.model_name,
                    processors=processors,
                )
                logger.info("Stanza loaded after download (sem NER)")
        except Exception as e2:
            logger.error(f"Failed to load Stanza: {e2}")
            raise
    
    def _ensure_model(self):
        if self.model is None:
            self.load_model()
    
    def tokenize(self, text: str) -> List[str]:
        self._ensure_model()
        doc = self.model(text)
        return [w.text for sent in doc.sentences for w in sent.words]
    
    def pos_tag(self, tokens: List[str]) -> List[str]:
        self._ensure_model()
        doc = self.model(" ".join(tokens))
        return [w.upos for sent in doc.sentences for w in sent.words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        self._ensure_model()
        doc = self.model(" ".join(tokens))
        return [w.lemma for sent in doc.sentences for w in sent.words]
    
    def dependency_parse(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        self._ensure_model()
        doc = self.model(" ".join(tokens))
        heads, deprels = [], []
        for sent in doc.sentences:
            for w in sent.words:
                heads.append(w.head)
                deprels.append(w.deprel)
        return heads, deprels
    
    def ner(self, tokens: List[str]) -> List[str]:
        """Português sem NER: retorna somente "O"."""
        return ["O"] * len(tokens)


# ==================== UDPIPE WRAPPER ====================

class UDPipeWrapper(NLPModelWrapper):
    """Wrapper para UDPipe."""
    
    def load_model(self):
        import ufal.udpipe as udpipe
        
        self.udpipe = udpipe
        self.model = udpipe.Model.load(self.model_name)
        
        if not self.model:
            raise RuntimeError(f"Failed to load UDPipe: {self.model_name}")
        
        self.pipeline = udpipe.Pipeline(
            self.model,
            "tokenize",
            udpipe.Pipeline.DEFAULT,
            udpipe.Pipeline.DEFAULT,
            "conllu",
        )
        logger.info(f"UDPipe loaded: {self.model_name}")
    
    def _ensure_model(self):
        if self.model is None:
            self.load_model()
    
    def _process(self, text: str) -> str:
        self._ensure_model()
        return self.pipeline.process(text)
    
    def _field(self, conllu: str, idx: int, as_int: bool = False):
        """Extrai campo específico do output CoNLL-U."""
        vals = []
        for line in conllu.strip().split("\n"):
            if line and not line.startswith("#") and line[0].isdigit():
                parts = line.split("\t")
                if "-" in parts[0]:
                    continue
                
                val = parts[idx] if idx < len(parts) else "_"
                
                if as_int:
                    try:
                        val = int(val)
                    except:
                        val = 0
                
                vals.append(val)
        
        return vals
    
    def tokenize(self, text: str) -> List[str]:
        conllu = self._process(text)
        return self._field(conllu, 1)
    
    def pos_tag(self, tokens: List[str]) -> List[str]:
        conllu = self._process(" ".join(tokens))
        return self._field(conllu, 3)
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        conllu = self._process(" ".join(tokens))
        return self._field(conllu, 2)
    
    def dependency_parse(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        conllu = self._process(" ".join(tokens))
        heads = self._field(conllu, 6, as_int=True)
        deprels = self._field(conllu, 7)
        return heads, deprels
    
    def ner(self, tokens: List[str]) -> List[str]:
        """UDPipe não suporta NER."""
        return ["O"] * len(tokens)


# ==================== FACTORY ====================

def get_model_wrapper(
    framework: str,
    model_name: str,
    device: str = "cpu"
) -> NLPModelWrapper:
    """Factory para criar wrappers."""
    
    f = framework.lower()
    
    if f == "spacy":
        return SpacyWrapper(model_name, device)
    elif f == "stanza":
        return StanzaWrapper(model_name, device)
    elif f == "udpipe":
        return UDPipeWrapper(model_name, device)
    else:
        raise ValueError(f"Unknown framework: {framework}")