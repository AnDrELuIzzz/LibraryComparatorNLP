from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
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


# ==================== SPACY WRAPPER - RECUPERADO E MELHORADO ====================

class SpacyWrapper(NLPModelWrapper):
    """
    Wrapper para spaCy com suporte a tokens gold (pré-tokenizados).
    """
    def load_model(self):
        """ADICIONE ESTE MÉTODO!"""
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
        """NER usando tokens gold."""
        doc = self._process_with_gold_tokens(tokens)
        tags = ["O"] * len(tokens)
        
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                if i < len(tags):
                    tags[i] = ("B-" if i == ent.start else "I-") + ent.label_
        
        return tags

# ==================== STANZA WRAPPER - RECUPERADO E CORRIGIDO ====================

class StanzaWrapper(NLPModelWrapper):
    """
    Wrapper para Stanza - RECUPERADO com FIX do erro torch.load.
    """
    
    def load_model(self):
        import stanza
        import torch
        import os
        
        # ✅ FIX CRÍTICO: Desabilita weights_only no PyTorch 2.6+
        os.environ['TORCH_WEIGHTS_ONLY'] = '0'
        
        # Backup: adiciona safe_globals
        try:
            torch.serialization.add_safe_globals([
                __import__('numpy').core.multiarray._reconstruct,
                __import__('numpy').ndarray,
            ])
        except:
            pass
        
        try:
            self.model = stanza.Pipeline(
                self.model_name,
                processors="tokenize,pos,lemma,depparse,ner",
                download_method=None
            )
            logger.info(f"Stanza loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Error: {e}. Trying download...")
            try:
                stanza.download(self.model_name)
                self.model = stanza.Pipeline(
                    self.model_name,
                    processors="tokenize,pos,lemma,depparse,ner"
                )
                logger.info(f"Stanza loaded after download")
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
        """NER do Stanza com alinhamento por caractere."""
        self._ensure_model()
        doc = self.model(" ".join(tokens))
        tags = ["O"] * len(tokens)
        
        # Mapeamento char -> token
        char_to_token = {}
        char_pos = 0
        for i, token in enumerate(tokens):
            for _ in range(len(token)):
                char_to_token[char_pos] = i
                char_pos += 1
            char_pos += 1  # espaço
        
        # Processa entidades
        for sent in doc.sentences:
            for ent in sent.ents:
                start_char = ent.start_char
                end_char = ent.end_char
                
                if start_char in char_to_token and (end_char - 1) in char_to_token:
                    start_token = char_to_token[start_char]
                    end_token = char_to_token[end_char - 1] + 1
                    
                    for i in range(start_token, min(end_token, len(tags))):
                        if tags[i] == "O":
                            tags[i] = ("B-" if i == start_token else "I-") + ent.type
        
        return tags


# ==================== HUGGINGFACE WRAPPER - RECUPERADO E MELHORADO ====================

class HuggingFaceNERWrapper(NLPModelWrapper):
    """
    Wrapper para NER da HuggingFace - RECUPERADO do seu código antigo com melhorias.
    
    Alinha entidades aos tokens originais com alinhamento por caractere (seu método original).
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        self.pipeline = None
    
    def load_model(self):
        from transformers import pipeline
        
        device_id = 0 if self.device == "cuda" and torch.cuda.is_available() else -1
        
        self.pipeline = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",
            device=device_id,
        )
        logger.info(f"HuggingFace NER loaded: {self.model_name}")
    
    def _ensure_model(self):
        if self.pipeline is None:
            self.load_model()
    
    def tokenize(self, text: str) -> List[str]:
        # Para NER, usamos tokenização "gold" (WikiNER)
        return text.split()
    
    def pos_tag(self, tokens: List[str]) -> List[str]:
        return ["X"] * len(tokens)
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        return tokens
    
    def dependency_parse(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        heads = [0] * len(tokens)
        deprels = ["dep"] * len(tokens)
        return heads, deprels
    
    def ner(self, tokens: List[str]) -> List[str]:
        """
        NER com alinhamento por caractere (seu método original - MANTIDO e MELHORADO).
        """
        self._ensure_model()
        
        text = " ".join(tokens)
        predictions = self.pipeline(text)
        
        return self._align_predictions_to_tokens(tokens, predictions)
    
    @staticmethod
    def _align_predictions_to_tokens(tokens: List[str], predictions: List[dict]) -> List[str]:
        """
        Alinhamento por caractere (seu método original adaptado).
        
        Mapeia spans de entidades aos índices de token, gerando tags IOB.
        """
        if not predictions:
            return ["O"] * len(tokens)
        
        text = " ".join(tokens)
        token_tags = ["O"] * len(tokens)
        
        # Mapa de caractere -> índice de token
        char_to_token_idx = []
        current_token = 0
        
        for token in tokens:
            for _ in range(len(token)):
                char_to_token_idx.append(current_token)
            
            # Espaço entre tokens
            char_to_token_idx.append(current_token)
            current_token += 1
        
        # Processa predições
        for pred in predictions:
            start = pred.get("start", 0)
            end = pred.get("end", 0)
            label = pred.get("entity_group", "MISC")
            
            if start >= len(char_to_token_idx) or end > len(char_to_token_idx):
                continue
            
            start_token_idx = char_to_token_idx[start]
            end_token_idx = char_to_token_idx[min(end - 1, len(char_to_token_idx) - 1)]
            
            # Marca B-/I- em IOB
            token_tags[start_token_idx] = f"B-{label}"
            for t in range(start_token_idx + 1, end_token_idx + 1):
                if t < len(token_tags):
                    token_tags[t] = f"I-{label}"
        
        return token_tags


# ==================== UDPIPE WRAPPER - RECUPERADO ====================

class UDPipeWrapper(NLPModelWrapper):
    """Wrapper para UDPipe - MANTIDO do código antigo."""
    
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
    elif f == "huggingface_ner":
        return HuggingFaceNERWrapper(model_name, device)
    elif f == "udpipe":
        return UDPipeWrapper(model_name, device)
    else:
        raise ValueError(f"Unknown framework: {framework}")