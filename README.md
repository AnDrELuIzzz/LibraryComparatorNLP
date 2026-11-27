# LibraryComparatorNLP

Código, dados, saídas e métricas do artigo **"Análise Comparativa de Ferramentas para
Processamento de Linguagem Natural"**: avaliação *off-the-shelf* de spaCy, Stanza e UDPipe
para o português, em tokenização, etiquetagem morfossintática, traços morfológicos,
lematização e análise de dependências (teste do UD Portuguese-Bosque 2.12) e em
reconhecimento de entidades nomeadas (WikiNER).

## Início rápido

Requer Python 3.12 (o artigo usou 3.12.5).

**Reproduzir os números do artigo** (a partir das saídas já salvas em `outputs/`; não roda
nenhum modelo, leva cerca de 1 minuto):

```bash
make install-light
make reproduce      # testes + results/reanalysis/{reanalysis.json,tabelas.md}
```

**Rodar tudo do zero** (executa as três ferramentas; baixa os modelos sozinho):

```bash
make install        # PyTorch CPU, spaCy + pt_core_news_lg, Stanza, UDPipe e modelos
make pipeline       # regenera outputs/ e results/
make test           # confere os resultados regenerados
```

Sem `make`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-reanalysis.txt && python -m pytest -q tests/ && python reanalysis.py
# pipeline completo:
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && python main.py
```

## Onde está cada resultado do artigo

| Artigo | Arquivo | Gerado por |
|---|---|---|
| Tabela 4 (UD oficial: P/R/F1/AlignedAcc) | `results/ud_eval/{spacy,stanza,udpipe}.txt` | `main.py` (`conll18_ud_eval.py -v`) |
| Tabela 5 (NER por entidade) e NER por token (Seção 4.3) | `results/ner/spacy.json` | `main.py` (seqeval / scikit-learn) |
| Tabela 6 (recall por subconjunto) e IC95% do bootstrap (Seção 4.4) | `results/reanalysis/tabelas.md` | `reanalysis.py` |
| Sobreposição treino/teste (Seção 3.2) | `results/reanalysis/tabelas.md` | `reanalysis.py` |
| Palavras alinhadas pelas três ferramentas (Seções 4.1.2 e 5.1) | `results/reanalysis/tabelas.md` | `reanalysis.py` |
| Lemas sem distinção de caixa (Seção 4.1.3) | `results/reanalysis/tabelas.md` | `reanalysis.py` |
| BI, MK e MCC por classe UPOS (Seção 4.2) | `results/reanalysis/tabelas.md` | `reanalysis.py` |
| Versões do ambiente (Tabela 2) | `results/full_results.json` → `metadata.versions` | `main.py` |

Os mesmos números estão em formato de máquina em `results/reanalysis/reanalysis.json`.
`tests/test_reanalysis.py` confere, entre outros pontos, que:

- `tools/conll18_ud_eval.py` é o script oficial v1.2 sem alterações (md5);
- o recall do subconjunto "total" reproduz a coluna Recall de `results/ud_eval/*.txt`;
- a sobreposição do teste com as partições 2.5/2.8 é de 768 / 288 / 111 sentenças;
- o NER do spaCy dá F1 por entidade 0,9385 e por token 0,9336 (gold IOB2);
- a diferença UDPipe − Stanza em LAS nas 111 inéditas é −4,14, com IC95% abaixo de zero.

## Estrutura

```
main.py                  pipeline completo (modelos → avaliação → relatório → reanálise)
reanalysis.py            análises complementares a partir das saídas salvas
config.json              caminhos e modelos (caminhos relativos à raiz do repositório)
src/
  model_wrapper.py       adaptadores spaCy / Stanza / UDPipe
  run_pipeline.py        texto bruto → CoNLL-U (com MWT e normalização das árvores); NER no WikiNER
  evaluate_ud_official.py  roda tools/conll18_ud_eval.py
  evaluate_wikiner.py    NER por token e por entidade (seqeval)
  ud_alignment.py        reaproveita o alinhamento do script oficial sem modificá-lo
  bosque_overlap.py      sobreposição do teste com partições de versões anteriores
  subset_eval.py         métricas por subconjunto, interseção, lemas sem caixa, contrações
  paired_bootstrap.py    bootstrap pareado por sentenças (B = 10.000, semente 42)
  informedness_markedness.py  BI, MK e MCC por classe
  resources.py           download e conferência (md5) dos modelos
scripts/
  download_models.py     baixa os modelos (main.py faz isso automaticamente)
  download_ud_bosque.py  regenera data/bosque_test_partition_origin.tsv a partir do UD
tools/conll18_ud_eval.py script oficial da CoNLL 2018, v1.2
data/                    teste do Bosque, WikiNER e origem das sentenças do teste
outputs/                 saídas das ferramentas (CoNLL-U e BIO)
results/                 métricas
tests/                   testes de consistência
```

## Ambiente e modelos

| Item | Versão |
|---|---|
| Python | 3.12.5 |
| spaCy / pt_core_news_lg | 3.8.11 / 3.8.0 |
| Stanza (biblioteca) / modelos (pacote padrão `pt`, recursos) | 1.11.0 / **1.8.0** |
| PyTorch (CPU) | 2.9.1 |
| ufal.udpipe / modelo | 1.4.0.1 / `portuguese-bosque-ud-2.5-191206.udpipe` |
| seqeval | 1.2.2 |
| scikit-learn / NumPy | 1.7.2 / 2.3.5 |

O modelo do UDPipe é o do pacote UD 2.5 (Straka; Straková, 2019;
<http://hdl.handle.net/11234/1-3131>). `src/resources.py` o baixa do espelho usado pelo
pacote R `udpipe` e confere o md5 (`a6d426f6e49b95f8425bc5cc546c4c77`), idêntico ao do LINDAT.
**Atenção ao Stanza:** a versão dos modelos é independente da versão da biblioteca. Os números
do artigo foram obtidos com a biblioteca 1.11.0 e os modelos dos **recursos 1.8.0** (processadores
`tokenize,mwt,pos,lemma,depparse`, todos treinados no Bosque). `config.json` fixa essa versão e
`src/resources.py` baixa os modelos para `models/stanza/`, fora do `~/stanza_resources`
compartilhado. Com os modelos 1.11.0, a saída muda (LAS 88,33% em vez de 87,43%).
`make pipeline` reproduz byte a byte todos os arquivos de `outputs/` e as métricas de `results/`.

## Dados

- `data/pt_bosque-ud-test.conllu`: teste do UD Portuguese-Bosque 2.12 (idêntico nas versões
  2.12 a 2.17); 1.167 sentenças, 27.604 palavras sintáticas, 2.015 tokens multipalavra;
  521 sentenças do CETENFolha (PB) e 646 do CETEMPúblico (PE).
- `data/bosque_test_partition_origin.tsv`: para cada `sent_id` do teste, a variedade e a
  partição de origem nas versões 2.5, 2.8, 2.12 e 2.15. Regenerável com
  `python scripts/download_ud_bosque.py`.
- `data/aij-wikiner-pt-wp3`: WikiNER (português). São avaliadas as primeiras 5.000 linhas
  (4.701 sentenças), com gold no esquema IOB1.

## Decisões de implementação que afetam as métricas

- **Contrações (spaCy).** O spaCy não gera tokens multipalavra; contrações como *do* e *pelo*
  ficam como palavra única, o que reduz o F1 das métricas UD do spaCy
  (`unexpanded_contractions` em `src/subset_eval.py`).
- **Lemas em minúsculas.** `src/model_wrapper.py` grava os lemas das três ferramentas em
  minúsculas; o script oficial compara lemas com distinção de caixa. `reanalysis.py` reporta
  também Lemmas/BLEX sem distinção de caixa.
- **Normalização das árvores.** `src/run_pipeline.py` reanexa raízes adicionais à primeira
  raiz (relação `dep`) e desfaz ciclos (`normalize_ud_tree`). XPOS é gravado como `_`.
- **Alinhamento.** As análises por palavra usam o alinhamento do próprio
  `conll18_ud_eval.py`, capturado sem modificar o script (`src/ud_alignment.py`).
- **NER.** Só o spaCy tem modelo de NER para o português; Stanza e UDPipe não são avaliados.
- **Sobreposição treino/teste.** O UDPipe (UD 2.5) e o spaCy (Bosque v2.8) foram treinados em
  versões anteriores do Bosque, cujas partições diferem das atuais: 768 sentenças do teste
  estavam no treino dessas versões. Os resultados são reportados também por subconjunto.

## Licença

MIT (ver `LICENSE`).
