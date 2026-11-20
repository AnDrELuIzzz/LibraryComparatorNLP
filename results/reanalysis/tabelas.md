# Reanálise (gerado por reanalysis.py)

## Sobreposição do teste (UD 2.12) com as partições anteriores

| Versão | treino | desenvolvimento | teste | ausente |
|---|---|---|---|---|
| 2.5 | 768 (65,80%) | 288 (24,70%) | 111 (9,50%) | 0 |
| 2.8 | 768 (65,80%) | 288 (24,70%) | 111 (9,50%) | 0 |
| 2.12 | 0 | 0 | 1167 (100,00%) | 0 |
| 2.15 | 0 | 0 | 1167 (100,00%) | 0 |

Variedade × partição 2.5: {'PB_dev': 288, 'PB_test': 111, 'PB_train': 122, 'PE_train': 646}

## Recall (%) por subconjunto (Tabela 7 do artigo)

| Subconjunto (sentenças) | spaCy UPOS / UAS / LAS | Stanza UPOS / UAS / LAS | UDPipe UPOS / UAS / LAS |
|---|---|---|---|
| Treino (768) | 83,80 / 74,78 / 72,58 | 97,70 / 90,53 / 86,97 | 99,24 / 95,43 / 94,55 |
| Desenvolvimento (288) | 83,93 / 75,45 / 72,34 | 97,52 / 91,65 / 88,14 | 96,28 / 87,08 / 83,58 |
| Teste, inéditas (111) | 81,84 / 74,70 / 72,21 | 97,13 / 92,77 / 89,68 | 96,12 / 88,85 / 85,54 |
| Total (1167) | 83,66 / 74,91 / 72,50 | 97,61 / 90,95 / 87,44 | 98,37 / 93,15 / 91,52 |

## Bootstrap pareado (diferença em pontos; IC95% por percentis)

| Comparação | Subconjunto | Métrica | A | B | A − B | IC95% |
|---|---|---|---|---|---|---|
| udpipe-stanza | total | UPOS | 98,37 | 97,61 | +0,76 | [+0,53; +0,99] |
| udpipe-stanza | total | UAS | 93,15 | 90,95 | +2,21 | [+1,58; +2,82] |
| udpipe-stanza | total | LAS | 91,52 | 87,44 | +4,09 | [+3,38; +4,78] |
| udpipe-stanza | ineditas | UPOS | 96,12 | 97,13 | -1,00 | [-1,68; -0,37] |
| udpipe-stanza | ineditas | UAS | 88,85 | 92,77 | -3,92 | [-5,45; -2,48] |
| udpipe-stanza | ineditas | LAS | 85,54 | 89,68 | -4,14 | [-5,98; -2,44] |
| udpipe-stanza | fora_do_treino | UPOS | 96,24 | 97,41 | -1,17 | [-1,57; -0,78] |
| udpipe-stanza | fora_do_treino | UAS | 87,58 | 91,97 | -4,39 | [-5,38; -3,40] |
| udpipe-stanza | fora_do_treino | LAS | 84,14 | 88,58 | -4,44 | [-5,50; -3,40] |
| udpipe-stanza | treino | UPOS | 99,24 | 97,70 | +1,55 | [+1,29; +1,81] |
| udpipe-stanza | treino | UAS | 95,43 | 90,53 | +4,91 | [+4,24; +5,56] |
| udpipe-stanza | treino | LAS | 94,55 | 86,97 | +7,58 | [+6,83; +8,30] |
| stanza-spacy | total | UPOS | 97,61 | 83,66 | +13,95 | [+13,35; +14,57] |
| stanza-spacy | total | UAS | 90,95 | 74,91 | +16,04 | [+15,25; +16,82] |
| stanza-spacy | total | LAS | 87,44 | 72,50 | +14,94 | [+14,15; +15,74] |
| stanza-spacy | ineditas | UPOS | 97,13 | 81,84 | +15,29 | [+13,23; +17,35] |
| stanza-spacy | ineditas | UAS | 92,77 | 74,70 | +18,07 | [+15,82; +20,31] |
| stanza-spacy | ineditas | LAS | 89,68 | 72,21 | +17,47 | [+15,02; +19,90] |
| stanza-spacy | fora_do_treino | UPOS | 97,41 | 83,33 | +14,07 | [+12,96; +15,15] |
| stanza-spacy | fora_do_treino | UAS | 91,97 | 75,23 | +16,74 | [+15,48; +17,96] |
| stanza-spacy | fora_do_treino | LAS | 88,58 | 72,30 | +16,28 | [+15,00; +17,52] |
| stanza-spacy | treino | UPOS | 97,70 | 83,80 | +13,90 | [+13,15; +14,65] |
| stanza-spacy | treino | UAS | 90,53 | 74,78 | +15,75 | [+14,75; +16,76] |
| stanza-spacy | treino | LAS | 86,97 | 72,58 | +14,39 | [+13,38; +15,41] |
| udpipe-spacy | total | UPOS | 98,37 | 83,66 | +14,71 | [+14,09; +15,32] |
| udpipe-spacy | total | UAS | 93,15 | 74,91 | +18,24 | [+17,40; +19,08] |
| udpipe-spacy | total | LAS | 91,52 | 72,50 | +19,03 | [+18,15; +19,89] |
| udpipe-spacy | ineditas | UPOS | 96,12 | 81,84 | +14,29 | [+12,15; +16,40] |
| udpipe-spacy | ineditas | UAS | 88,85 | 74,70 | +14,16 | [+11,46; +16,70] |
| udpipe-spacy | ineditas | LAS | 85,54 | 72,21 | +13,33 | [+10,60; +15,94] |
| udpipe-spacy | fora_do_treino | UPOS | 96,24 | 83,33 | +12,90 | [+11,73; +14,03] |
| udpipe-spacy | fora_do_treino | UAS | 87,58 | 75,23 | +12,35 | [+10,87; +13,85] |
| udpipe-spacy | fora_do_treino | LAS | 84,14 | 72,30 | +11,84 | [+10,35; +13,34] |
| udpipe-spacy | treino | UPOS | 99,24 | 83,80 | +15,45 | [+14,72; +16,19] |
| udpipe-spacy | treino | UAS | 95,43 | 74,78 | +20,66 | [+19,69; +21,62] |
| udpipe-spacy | treino | LAS | 94,55 | 72,58 | +21,97 | [+20,99; +22,94] |

## Acurácia sobre as 23483 palavras alinhadas pelas três ferramentas

| Ferramenta | UPOS | UFeats | Lemas (sem caixa) | UAS | LAS |
|---|---|---|---|---|---|
| spacy | 98,13 | 97,00 | 98,64 | 87,89 | 85,09 |
| stanza | 97,52 | 96,68 | 98,65 | 90,02 | 86,03 |
| udpipe | 98,28 | 97,23 | 99,06 | 92,36 | 90,52 |

## Lemas sem distinção entre maiúsculas e minúsculas

| Ferramenta | Lemmas F1 | Lemmas AlignedAcc | BLEX F1 |
|---|---|---|---|
| spacy | 79,51 → 87,29 | 89,83 → 98,62 | 66,41 → 79,22 |
| stanza | 91,13 → 98,56 | 91,34 → 98,79 | 69,45 → 80,76 |
| udpipe | 91,50 → 98,99 | 91,65 → 99,15 | 74,46 → 87,78 |

Linhas de tokens multipalavra nos arquivos preditos: {'spacy': 0, 'stanza': 2018, 'udpipe': 2034}
Tokens multipalavra da referência deixados sem expansão: {'spacy': {'gold_multiword_tokens': 2015, 'unexpanded': 2015}, 'stanza': {'gold_multiword_tokens': 2015, 'unexpanded': 7}, 'udpipe': {'gold_multiword_tokens': 2015, 'unexpanded': 0}}

## BI e MK por classe UPOS sobre palavras alinhadas (Tabela 5 do artigo)

| Classe | spaCy BI | spaCy MK | Stanza BI | Stanza MK | UDPipe BI | UDPipe MK | n (Sp / St / UD) |
|---|---|---|---|---|---|---|---|
| ADJ | 0,956 | 0,970 | 0,934 | 0,954 | 0,959 | 0,968 | 1404 / 1404 / 1407 |
| ADP | 0,987 | 0,983 | 0,989 | 0,987 | 0,993 | 0,992 | 2205 / 4105 / 4111 |
| ADV | 0,971 | 0,969 | 0,953 | 0,971 | 0,979 | 0,974 | 1029 / 1033 / 1033 |
| AUX | 0,984 | 0,960 | 0,984 | 0,946 | 0,990 | 0,969 | 662 / 663 / 663 |
| CCONJ | 0,994 | 0,992 | 0,995 | 0,989 | 0,997 | 0,992 | 642 / 642 / 642 |
| DET | 0,991 | 0,987 | 0,992 | 0,988 | 0,995 | 0,992 | 2329 / 4201 / 4203 |
| INTJ | 0,000 | 0,000 | 0,000 | 0,000 | 0,600 | 1,000 | 5 / 5 / 5 |
| NOUN | 0,977 | 0,980 | 0,973 | 0,970 | 0,979 | 0,980 | 5045 / 5042 / 5046 |
| NUM | 0,974 | 0,985 | 0,966 | 0,977 | 0,976 | 0,980 | 501 / 505 / 507 |
| PRON | 0,964 | 0,969 | 0,964 | 0,962 | 0,975 | 0,979 | 834 / 963 / 960 |
| PROPN | 0,972 | 0,958 | 0,955 | 0,958 | 0,970 | 0,962 | 2080 / 2076 / 2083 |
| PUNCT | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 3594 / 3595 / 3596 |
| SCONJ | 0,929 | 0,933 | 0,935 | 0,917 | 0,952 | 0,948 | 653 / 662 / 663 |
| SYM | 1,000 | 1,000 | 1,000 | 1,000 | 0,983 | 1,000 | 60 / 60 / 60 |
| VERB | 0,982 | 0,980 | 0,969 | 0,975 | 0,980 | 0,983 | 2460 / 2552 / 2551 |
| X | 0,412 | 0,999 | 0,412 | 0,666 | 0,353 | 0,999 | 34 / 34 / 34 |

## NER (spaCy, WikiNER)

- Sentenças: 4701; tokens: 121387
- Por token, gold IOB1 original (média ponderada, sem O): P=0.9467 R=0.3760 F1=0.5367
- Por token, gold convertido para IOB2: P=0.9400 R=0.9276 F1=0.9336
- Por entidade (seqeval): P=0.9372 R=0.9397 F1=0.9385
