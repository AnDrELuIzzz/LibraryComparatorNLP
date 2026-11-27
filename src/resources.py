# -*- coding: utf-8 -*-
"""Garante que os modelos usados no estudo estejam disponíveis, baixando-os se preciso.

- UDPipe 1: portuguese-bosque-ud-2.5-191206.udpipe (Straka; Straková, 2019), conferido por md5.
  Fonte oficial: LINDAT, http://hdl.handle.net/11234/1-3131. O download automático usa o
  espelho do pacote R ``udpipe`` (jwijffels/udpipe.models.ud.2.5), cujo arquivo é idêntico
  byte a byte ao do LINDAT; o md5 abaixo garante isso.
- spaCy: pt_core_news_lg (instalado via requirements.txt; baixado aqui se faltar).
- Stanza: pacote padrão ``pt`` dos recursos 1.8.0, em ``models/stanza`` (ver ensure_stanza_model).
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# md5 do default.zip do português por versão de recursos do Stanza (de resources_<versão>.json)
STANZA_DEFAULT_MD5 = {"1.8.0": "5a1038d1fa8c21d22659889f269d2adc"}

UDPIPE_MODEL_MD5 = "a6d426f6e49b95f8425bc5cc546c4c77"
UDPIPE_MODEL_URL = ("https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/master/inst/"
                    "udpipe-ud-2.5-191206/portuguese-bosque-ud-2.5-191206.udpipe")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_udpipe_model(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        logger.info(f"Baixando o modelo do UDPipe para {path} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(UDPIPE_MODEL_URL, tmp)
        os.replace(tmp, path)
    got = md5(path)
    if got != UDPIPE_MODEL_MD5:
        raise RuntimeError(f"md5 inesperado para {path}: {got} (esperado {UDPIPE_MODEL_MD5}). "
                           "Apague o arquivo e rode de novo.")
    return path


def ensure_spacy_model(name: str) -> None:
    import spacy.util
    if not spacy.util.is_package(name):
        from spacy.cli import download
        logger.info(f"Baixando o modelo do spaCy {name} ...")
        download(name)


def ensure_stanza_model(lang: str, model_dir: str, resources_version: str) -> None:
    """Baixa o pacote padrão de ``lang`` na versão de recursos fixada, num diretório próprio.

    A versão dos MODELOS do Stanza é independente da versão da biblioteca: os números do artigo
    foram obtidos com a biblioteca 1.11.0 e os modelos dos recursos 1.8.0. Sem fixar a versão
    e o diretório, ``stanza.download`` traz os modelos mais recentes para ~/stanza_resources e
    os resultados mudam (com os modelos 1.11.0, a LAS do Stanza passa de 87,43% para 88,33%).
    """
    import stanza
    if os.path.exists(os.path.join(model_dir, "resources.json")):
        import json
        with open(os.path.join(model_dir, "resources.json"), encoding="utf-8") as f:
            md5_ok = json.load(f).get(lang, {}).get("default_md5") == STANZA_DEFAULT_MD5.get(resources_version)
        if md5_ok and os.path.isdir(os.path.join(model_dir, lang, "depparse")):
            return
    logger.info(f"Baixando os modelos do Stanza ({lang}, recursos {resources_version}) para {model_dir} ...")
    stanza.download(lang, model_dir=model_dir, resources_version=resources_version, verbose=False)


def ensure_all(config: dict) -> None:
    ensure_udpipe_model(config["paths"]["udpipe_model_path"])
    ensure_spacy_model(config["models"]["spacy"])
    ensure_stanza_model(config["models"]["stanza_lang"], config["paths"]["stanza_dir"],
                        config["models"]["stanza_resources_version"])
