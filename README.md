# LibraryComparatorNLP

Avaliação *off-the-shelf* de spaCy, Stanza e UDPipe para o português: tokenização,
etiquetagem morfossintática, traços morfológicos, lematização e análise de dependências
no teste do UD Portuguese-Bosque, e reconhecimento de entidades nomeadas (NER) no WikiNER.

O repositório tem duas etapas:

1. **Pipeline** (`main.py`): executa os modelos, grava as saídas em `outputs/` e as
   avaliações oficiais em `results/`.
2. **Reanálise** (`reanalysis.py`): a partir das saídas **já salvas**, sem reexecutar
   modelos, calcula a sobreposição entre treino e teste, o recall por subconjunto de
   sentenças, o bootstrap pareado, BI/MK/MCC sobre palavras alinhadas, as métricas sobre
   as palavras alinhadas pelas três ferramentas, os lemas sem distinção de caixa e o NER
   por token com o gold convertido para IOB2. Resultados em `results/reanalysis/`.

## Reproduzir os números do artigo

```bash
pip install -r requirements-reanalysis.txt
python reanalysis.py          # ~20 s; gera results/reanalysis/reanalysis.json e tabelas.md
python -m pytest -q tests/    # confere a reanálise contra os resultados oficiais versionados
```

Os testes verificam, entre outros pontos, que:

- `tools/conll18_ud_eval.py` é o script oficial v1.2 sem alterações (md5);
- o recall do subconjunto "total" reproduz a coluna Recall de `results/ud_eval/*.txt` (Tabela 4);
- a sobreposição do teste com as partições 2.5/2.8 é de 768 / 288 / 111 sentenças;
- o NER por token com o gold IOB1 reproduz `results/ner/spacy.json`, e com o gold IOB2 dá F1 0,9336;
- o bootstrap é determinístico para a mesma semente.

| Artigo | Origem |
|---|---|
| Tabela 4 | `results/ud_eval/*.txt` (pipeline, `conll18_ud_eval.py -v`) |
| Tabela 5 (BI/MK por classe) | `reanalysis.py` → seção "BI e MK" de `tabelas.md` |
| Tabela 6 (NER por entidade) | `results/ner/spacy.json` (seqeval) |
| Tabela 7 (recall por subconjunto) e IC do bootstrap | `reanalysis.py` |
| Sobreposição, palavras alinhadas, lemas sem caixa, contrações | `reanalysis.py` |

## Pipeline completo

```bash
cp config.example.json config.json   # ajuste caminhos e o modelo do UDPipe
python main.py
```

Modelos: spaCy `pt_core_news_lg`; Stanza, pacote padrão `pt` (processadores
`tokenize,mwt,pos,lemma,depparse`); UDPipe 1 com `portuguese-bosque-ud-2.5-191206.udpipe`
(baixar do LINDAT e indicar o caminho em `config.json`). Versões do ambiente:
`src/version.py` / `results/report.txt`.

## Dados

- `data/pt_bosque-ud-test.conllu`: teste do UD Portuguese-Bosque, idêntico ao da versão
  2.12 (e às versões 2.12 a 2.17); 1.167 sentenças, 27.604 palavras sintáticas. Contém
  sentenças do CETENFolha (PB, 521) e do CETEMPúblico (PE, 646).
- `data/bosque_test_partition_origin.tsv`: para cada `sent_id` do teste, a variedade e a
  partição de origem (train/dev/test) nas versões 2.5, 2.8, 2.12 e 2.15. Para regenerar a
  partir das partições oficiais: `python scripts/download_ud_bosque.py`.
- `data/aij-wikiner-pt-wp3`: WikiNER (português). São avaliadas as primeiras 5.000 linhas
  (4.701 sentenças). O gold está no esquema IOB1.

## Decisões de implementação que afetam as métricas

- **Contrações (spaCy).** A conversão para CoNLL-U não gera linhas de tokens multipalavra
  para o spaCy; as contrações (*do*, *pelo*, …) ficam como palavra única, o que reduz o F1
  das métricas UD do spaCy. Ver `unexpanded_contractions` em `src/subset_eval.py`.
- **Lemas em minúsculas.** `src/model_wrapper.py` grava os lemas das três ferramentas em
  minúsculas; o script oficial compara lemas com distinção de caixa. `reanalysis.py`
  reporta também Lemmas/BLEX sem distinção de caixa.
- **Normalização das árvores.** `src/run_pipeline.py` reanexa raízes adicionais e desfaz
  ciclos antes da avaliação (`normalize_ud_tree`).
- **Alinhamento.** As análises por palavra usam o alinhamento do próprio
  `conll18_ud_eval.py`, capturado sem modificar o script (`src/ud_alignment.py`).
- **Sobreposição treino/teste.** O UDPipe (UD 2.5) e o spaCy (Bosque v2.8) foram treinados
  em versões anteriores do Bosque, cujas partições diferem das atuais: 768 sentenças do
  teste estavam no treino dessas versões. Os resultados são reportados também por subconjunto.

## Licença

Ver `LICENSE`.
