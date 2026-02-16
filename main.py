import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
from scipy import stats
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

from error_analysis import ErrorAnalyzer
from version import print_versions
from data_loader import DataLoader
from model_wrapper import get_model_wrapper
from evaluation import EvaluationMetrics, CrossValidation, StatisticalTests
from nlp_stat_test import (
    BootstrapSignificanceTest,
    BootstrapResult,
    format_results_table
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("NLP_Comparador")


class AcademicNLPEvaluator:
    """Avaliador completo para artigo científico com K-Fold e NLPStatTest."""

    def __init__(self, config: dict, output_dir: str = "results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().isoformat()
        self.versions = print_versions()
        self.results = {}
        self.error_analyzer = ErrorAnalyzer()

    def run_kfold_with_statistics(self, sentences: list, framework: str, model_name: str) -> dict:
        """Roda K-Fold e coleta métricas."""
        logger.info(f"[K-Fold] Iniciando {framework} ({model_name}) com {self.config['n_splits']} folds")

        n = len(sentences)
        splits = CrossValidation.kfold_split(n, self.config["n_splits"], self.config["seed"])
        wrapper = get_model_wrapper(framework, model_name)

        try:
            wrapper.load_model()
        except Exception as e:
            logger.error(f"Erro ao carregar {framework}: {e}")
            return self._empty_results()

        fold_results = {
            "pos_accuracy": [],
            "lemma_accuracy": [],
            "uas": [],
            "las": [],
            "token_precision": [],
            "token_recall": [],
            "token_f1": [],
        }

        for fold_id, (_, test_idx) in enumerate(splits, start=1):
            logger.info(f"  Fold {fold_id}/{self.config['n_splits']}")
            fold_metrics = {
                "pos_correct": 0,
                "pos_total": 0,
                "lemma_correct": 0,
                "lemma_total": 0,
                "uas_correct": 0,
                "las_correct": 0,
                "deps_total": 0,
                "token_tp": 0,
                "token_gold_total": 0,
                "token_pred_total": 0,
            }

            for i in tqdm(test_idx, desc=f"Fold {fold_id}", leave=False):
                s = sentences[i]
                text = s["text"]
                gold_tokens = s["tokens"]
                gold_pos = s["pos"]
                gold_lemmas = s["lemmas"]
                gold_heads = s["heads"]
                gold_deprels = s["deprels"]

                try:
                    # ✅ USA TOKENS GOLD DIRETAMENTE - não retokeniza
                    pred_pos = wrapper.pos_tag(gold_tokens)
                    pred_lemmas = wrapper.lemmatize(gold_tokens)
                    pred_heads, pred_deprels = wrapper.dependency_parse(gold_tokens)
                    # Tokenização só para métricas de token
                    pred_tokens = wrapper.tokenize(text)
                except Exception as e:
                    logger.warning(f"Erro processando sentença {i}: {e}")
                    continue

                # Alinhamento - mesmo tamanho
                length = len(gold_tokens)
                gold_pos_align = gold_pos[:length]
                gold_lemmas_align = gold_lemmas[:length]
                gold_heads_align = gold_heads[:length]
                gold_deprels_align = gold_deprels[:length]

                pred_pos_align = pred_pos[:length]
                pred_lemmas_align = pred_lemmas[:length]
                pred_heads_align = pred_heads[:length]
                pred_deprels_align = pred_deprels[:length]

                # Coletar erros
                model_key = f"{framework}_{model_name}"
                self.error_analyzer.collect_errors_from_prediction(
                    sent_id=i,
                    text=text,
                    gold_tokens=gold_tokens,
                    gold_pos=gold_pos_align,
                    gold_lemmas=gold_lemmas_align,
                    gold_heads=gold_heads_align,
                    gold_deprels=gold_deprels_align,
                    pred_pos=pred_pos_align,
                    pred_lemmas=pred_lemmas_align,
                    pred_heads=pred_heads_align,
                    pred_deprels=pred_deprels_align,
                    model_name=model_key
                )

                # POS Accuracy
                pos_matches = sum(1 for g, p in zip(gold_pos_align, pred_pos_align) if g == p)
                fold_metrics["pos_correct"] += pos_matches
                fold_metrics["pos_total"] += len(gold_pos_align)

                # Lemma Accuracy
                lemma_matches = sum(1 for g, p in zip(gold_lemmas_align, pred_lemmas_align) if g == p)
                fold_metrics["lemma_correct"] += lemma_matches
                fold_metrics["lemma_total"] += len(gold_lemmas_align)

                # Dependency (UAS/LAS)
                uas_matches = sum(1 for g, p in zip(gold_heads_align, pred_heads_align) if g == p)
                las_matches = sum(
                    1 for gh, gl, ph, pl in
                    zip(gold_heads_align, gold_deprels_align, pred_heads_align, pred_deprels_align)
                    if gh == ph and gl == pl
                )

                fold_metrics["uas_correct"] += uas_matches
                fold_metrics["las_correct"] += las_matches
                fold_metrics["deps_total"] += len(gold_heads_align)

                # Tokenização
                token_tp = sum(1 for g, p in zip(gold_tokens, pred_tokens) if g == p)
                fold_metrics["token_tp"] += token_tp
                fold_metrics["token_gold_total"] += len(gold_tokens)
                fold_metrics["token_pred_total"] += len(pred_tokens)

            # Agrupa por fold
            if fold_metrics["pos_total"]:
                fold_results["pos_accuracy"].append(fold_metrics["pos_correct"] / fold_metrics["pos_total"])
            if fold_metrics["lemma_total"]:
                fold_results["lemma_accuracy"].append(fold_metrics["lemma_correct"] / fold_metrics["lemma_total"])
            if fold_metrics["deps_total"]:
                fold_results["uas"].append(fold_metrics["uas_correct"] / fold_metrics["deps_total"])
                fold_results["las"].append(fold_metrics["las_correct"] / fold_metrics["deps_total"])

            token_pred_total = fold_metrics["token_pred_total"]
            token_gold_total = fold_metrics["token_gold_total"]
            if token_pred_total and token_gold_total:
                precision = fold_metrics["token_tp"] / token_pred_total if token_pred_total else 0.0
                recall = fold_metrics["token_tp"] / token_gold_total if token_gold_total else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                fold_results["token_precision"].append(precision)
                fold_results["token_recall"].append(recall)
                fold_results["token_f1"].append(f1)

        return {
            "fold_scores": fold_results,
        }

    def run_ner_evaluation(self, wikiner_samples: list, model_name: str) -> dict:
        """
        Avalia NER usando spaCy com anotações gold do WikiNER.

        Args:
            wikiner_samples: Lista de dicionários com 'tokens' e 'tags' (formato WikiNER)
            model_name: Nome do modelo spaCy

        Returns:
            Dict com métricas NER (precision, recall, f1, per_class)
        """
        logger.info(f"[NER] Avaliando spaCy ({model_name})")

        try:
            wrapper = get_model_wrapper("spacy", model_name)
            wrapper.load_model()
        except Exception as e:
            logger.error(f"Erro ao carregar NER: {e}")
            return self._empty_ner_results()

        all_gold, all_pred = [], []
        ner_by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for idx, s in enumerate(tqdm(wikiner_samples, desc="NER", leave=False)):
            tokens = s["tokens"]
            gold = s["tags"]

            try:
                # Usa tokens gold para NER
                pred = wrapper.ner(tokens)
            except Exception as e:
                logger.warning(f"Erro em NER: {e}")
                continue

            length = min(len(gold), len(pred))
            all_gold.extend(gold[:length])
            all_pred.extend(pred[:length])

            # Coletar erros de NER
            self.error_analyzer.collect_ner_errors(
                tokens=tokens,
                gold_tags=gold[:length],
                pred_tags=pred[:length],
                model_name=f"spacy_ner_{model_name}",
                sent_id=idx
            )

            # Métricas por classe
            for g, p in zip(gold[:length], pred[:length]):
                # Extrai tipo de entidade (remove B-/I-)
                entity_gold = g.split("-")[-1] if "-" in g else g
                entity_pred = p.split("-")[-1] if "-" in p else p

                if g == p and g != "O":
                    # True Positive
                    ner_by_class[entity_gold]["tp"] += 1
                elif g != "O" and p == "O":
                    # False Negative
                    ner_by_class[entity_gold]["fn"] += 1
                elif g == "O" and p != "O":
                    # False Positive
                    ner_by_class[entity_pred]["fp"] += 1
                elif g != "O" and p != "O" and entity_gold != entity_pred:
                    # Erro de tipo
                    ner_by_class[entity_gold]["fn"] += 1
                    ner_by_class[entity_pred]["fp"] += 1

        # Métricas globais
        metrics = EvaluationMetrics.compute_ner_metrics(all_gold, all_pred)

        # Métricas por classe
        metrics["per_class"] = {}
        for entity_class, counts in ner_by_class.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            metrics["per_class"][entity_class] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": tp + fn
            }

        return metrics

    def compute_statistics(self, model_scores: dict) -> dict:
        """Calcula média, desvio padrão e IC."""
        stats_result = {}

        for metric_name, scores in model_scores.items():
            if isinstance(scores, list) and len(scores) > 0:
                scores_arr = np.array(scores)
                mean = np.mean(scores_arr)
                std = np.std(scores_arr, ddof=1) if len(scores_arr) > 1 else 0
                ci = 1.96 * std / np.sqrt(len(scores_arr))

                stats_result[metric_name] = {
                    "mean": float(mean),
                    "std": float(std),
                    "ci_lower": float(mean - ci),
                    "ci_upper": float(mean + ci),
                    "scores": [float(s) for s in scores]
                }

        return stats_result

    # ========================================================================
    # MÉTODOS COM NLPStatTest
    # ========================================================================

    def run_bootstrap_comparisons(
        self,
        all_stats: dict,
        metrics_to_compare: list = None,
        bootstrap_rounds: int = 10000
    ) -> dict:
        """
        Executa comparações pairwise com bootstrap automático (NLPStatTest).
        """
        if metrics_to_compare is None:
            metrics_to_compare = ["pos_accuracy", "uas", "las", "lemma_accuracy"]

        comparisons = {}
        model_keys = list(all_stats.keys())

        for metric in metrics_to_compare:
            logger.info(f"\n[NLPStatTest Bootstrap] Comparando {metric}...")
            comparisons[metric] = {}

            systems_scores = {}
            for model_key in model_keys:
                if metric in all_stats[model_key]:
                    systems_scores[model_key] = np.array(
                        all_stats[model_key][metric]["scores"]
                    )

            if len(systems_scores) < 2:
                logger.warning(f"Métrica {metric} não possui dados suficientes")
                continue

            # Executar comparações pairwise com NLPStatTest
            results = BootstrapSignificanceTest.compare_multiple_systems(
                systems_scores=systems_scores,
                metric_name=metric,
                n_bootstrap_rounds=bootstrap_rounds,
                alpha=0.05,
                seed=self.config["seed"]
            )

            # Armazenar resultados
            for comp_name, result in results.items():
                comparisons[metric][comp_name] = {
                    "p_value": result.p_value,
                    "mean_diff": result.observed_diff,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "significant": result.significant,
                    "bootstrap_rounds": result.n_bootstrap_rounds,
                }

                # Log
                if result.significant:
                    logger.info(f"  ✓ {comp_name}: p={result.p_value:.4f} [significativo]")
                else:
                    logger.info(f"  ✗ {comp_name}: p={result.p_value:.4f}")

        return comparisons

    def generate_academic_report(self, all_results: dict, all_stats: dict, comparisons: dict):
        """Gera relatório científico em TXT, JSON, CSV com análise de erros."""

        # === SALVAR ANÁLISE DE ERROS ===
        logger.info("\nGerando análise detalhada de erros...")

        # CSV de erros
        self.error_analyzer.save_to_csv(self.output_dir)
        logger.info(f"✓ Arquivos CSV de erros salvos em {self.output_dir}/")

        # Relatório detalhado de erros
        error_report_path = os.path.join(self.output_dir, "analise_erros_detalhada.txt")
        self.error_analyzer.save_detailed_report(error_report_path)
        logger.info(f"✓ Relatório de erros salvo em {error_report_path}")

        # Estatísticas de erros em JSON
        error_stats_path = os.path.join(self.output_dir, "estatisticas_erros.json")
        self.error_analyzer.save_statistics_json(error_stats_path)
        logger.info(f"✓ Estatísticas de erros salvas em {error_stats_path}")

        # === RELATÓRIO TXT COM MELHORIAS ===
        txt_path = os.path.join(self.output_dir, "relatorio_academico.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write("RELATÓRIO COMPLETO - AVALIAÇÃO COMPARATIVA DE FERRAMENTAS DE PLN\n")
            f.write("(Com Testes Estatísticos via NLPStatTest Bootstrap e Análise de Erros)\n")
            f.write("=" * 120 + "\n\n")

            # Metadados
            f.write("METADADOS\n")
            f.write("-" * 120 + "\n")
            f.write(f"Data/Hora: {self.timestamp}\n")
            f.write(f"K-Folds: {self.config['n_splits']}\n")
            f.write(f"Bootstrap Rounds (NLPStatTest): {self.config.get('bootstrap_rounds', 10000)}\n")
            f.write(f"Random Seed: {self.config['seed']}\n")
            f.write(f"Nível de Significância (α): 0.05\n\n")

            # Tabela de resumo MELHORADA
            f.write("RESUMO DE RESULTADOS (Média ± Desvio Padrão) [IC 95%]\n")
            f.write("-" * 120 + "\n")
            table_data = []

            for model_key, stats_dict in all_stats.items():
                row = [model_key]
                for metric in [
                    "pos_accuracy",
                    "lemma_accuracy",
                    "uas",
                    "las",
                    "token_precision",
                    "token_recall",
                    "token_f1",
                ]:
                    if metric in stats_dict:
                        mean = stats_dict[metric]["mean"]
                        std = stats_dict[metric]["std"]
                        ci_lower = stats_dict[metric].get("ci_lower", mean)
                        ci_upper = stats_dict[metric].get("ci_upper", mean)
                        row.append(f"{mean:.4f} ± {std:.4f}\n[{ci_lower:.4f}, {ci_upper:.4f}]")
                    else:
                        row.append("N/A")
                table_data.append(row)

            headers = [
                "Modelo",
                "POS Acc",
                "Lemma Acc",
                "UAS",
                "LAS",
                "Tok Prec",
                "Tok Rec",
                "Tok F1",
            ]

            f.write(tabulate(table_data, headers=headers, tablefmt="grid") + "\n\n")

            # TABELA DETALHADA POR MÉTRICA
            f.write("\n" + "="*120 + "\n")
            f.write("DETALHAMENTO POR MÉTRICA\n")
            f.write("="*120 + "\n\n")

            for metric in ["pos_accuracy", "lemma_accuracy", "uas", "las"]:
                f.write(f"\n{metric.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n")
                metric_table = []
                for model_key, stats_dict in all_stats.items():
                    if metric in stats_dict:
                        s = stats_dict[metric]
                        metric_table.append([
                            model_key,
                            f"{s['mean']:.6f}",
                            f"{s['std']:.6f}",
                            f"[{s['ci_lower']:.6f}, {s['ci_upper']:.6f}]",
                            ", ".join([f"{score:.4f}" for score in s["scores"]])
                        ])

                if metric_table:
                    f.write(tabulate(
                        metric_table,
                        headers=["Modelo", "Média", "Desvio Padrão", "IC 95%", "Scores por Fold"],
                        tablefmt="grid"
                    ) + "\n")

            # NER
            ner_rows = []
            for model_key, stats_dict in all_stats.items():
                if all(metric in stats_dict for metric in ["precision", "recall", "f1"]):
                    ner_rows.append([
                        model_key,
                        f"{stats_dict['precision']['mean']:.4f}",
                        f"{stats_dict['recall']['mean']:.4f}",
                        f"{stats_dict['f1']['mean']:.4f}",
                    ])

            if ner_rows:
                f.write("\n" + "="*120 + "\n")
                f.write("MÉTRICAS DE NER (Precisão, Revocação, F1)\n")
                f.write("="*120 + "\n")
                ner_headers = ["Modelo", "Precisão", "Revocação", "F1"]
                f.write(tabulate(ner_rows, headers=ner_headers, tablefmt="grid") + "\n\n")

                # Adicionar métricas por classe se disponível
                for model_key, result in all_results.items():
                    if "ner" in result and "per_class" in result["ner"]:
                        f.write(f"\nMétricas por classe - {model_key}:\n")
                        f.write("-" * 80 + "\n")
                        class_table = []
                        for entity_class, metrics in result["ner"]["per_class"].items():
                            class_table.append([
                                entity_class,
                                f"{metrics['precision']:.4f}",
                                f"{metrics['recall']:.4f}",
                                f"{metrics['f1']:.4f}",
                                metrics['support']
                            ])
                        f.write(tabulate(
                            class_table,
                            headers=["Classe", "Precisão", "Revocação", "F1", "Suporte"],
                            tablefmt="grid"
                        ) + "\n\n")

            # ESTATÍSTICAS DE ERROS
            error_stats = self.error_analyzer.get_error_statistics()
            f.write("\n" + "="*120 + "\n")
            f.write("RESUMO DE ANÁLISE DE ERROS\n")
            f.write("="*120 + "\n\n")

            f.write(f"Total de erros de POS Tagging: {error_stats['pos_errors']['total']}\n")
            if error_stats['pos_errors']['most_common']:
                f.write("  Erros mais comuns:\n")
                for error_type, count in error_stats['pos_errors']['most_common'][:5]:
                    f.write(f"    - {error_type}: {count}\n")

            f.write(f"\nTotal de erros de Lematização: {error_stats['lemma_errors']['total']}\n")
            if error_stats['lemma_errors']['by_pos']:
                f.write("  Por classe POS:\n")
                for pos, count in error_stats['lemma_errors']['by_pos'][:5]:
                    f.write(f"    - {pos}: {count}\n")

            f.write(f"\nTotal de erros de Dependências: {error_stats['dep_errors']['total']}\n")
            f.write(f"  - Apenas HEAD incorreto: {error_stats['dep_errors']['head_only_errors']}\n")
            f.write(f"  - Apenas DEPREL incorreto: {error_stats['dep_errors']['deprel_only_errors']}\n")
            f.write(f"  - Ambos incorretos: {error_stats['dep_errors']['both_errors']}\n")

            if error_stats['ner_errors']['total'] > 0:
                f.write(f"\nTotal de erros de NER: {error_stats['ner_errors']['total']}\n")
                if error_stats['ner_errors']['most_common']:
                    f.write("  Erros mais comuns:\n")
                    for error_type, count in error_stats['ner_errors']['most_common'][:5]:
                        f.write(f"    - {error_type}: {count}\n")

            f.write(f"\n→ Análise detalhada disponível em: analise_erros_detalhada.txt\n")
            f.write(f"→ Arquivos CSV: pos_errors.csv, lemma_errors.csv, dependency_errors.csv, ner_errors.csv\n")

            # Testes estatísticos
            f.write("\n\n" + "="*120 + "\n")
            f.write("TESTES ESTATÍSTICOS (NLPStatTest Bootstrap, p < 0.05 = significativo)\n")
            f.write("="*120 + "\n")

            for metric_name, metric_comparisons in comparisons.items():
                f.write(f"\n{metric_name.upper()}:\n")
                for comp_name, result in metric_comparisons.items():
                    if isinstance(result, dict) and result.get("p_value"):
                        p_val = result["p_value"]
                        mean_diff = result["mean_diff"]
                        ci_lower = result.get("ci_lower", "N/A")
                        ci_upper = result.get("ci_upper", "N/A")
                        sig = "✓ Significativo" if p_val < 0.05 else "✗ Não significativo"

                        if isinstance(ci_lower, float):
                            f.write(
                                f"  {comp_name}:\n"
                                f"    p-value = {p_val:.4f}\n"
                                f"    Δμ = {mean_diff:+.6f}\n"
                                f"    IC 95% = [{ci_lower:.6f}, {ci_upper:.6f}]\n"
                                f"    {sig}\n\n"
                            )
                        else:
                            f.write(f"  {comp_name}: p={p_val:.4f}, Δμ={mean_diff:+.4f} {sig}\n")

            f.write("\n" + "="*120 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*120 + "\n")

        logger.info(f"Relatório TXT salvo em {txt_path}")

        # === JSON DETALHADO ===
        json_path = os.path.join(self.output_dir, "metricas_completas.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json_data = {
                "metadata": {
                    "timestamp": self.timestamp,
                    "n_splits": self.config["n_splits"],
                    "seed": self.config["seed"],
                    "bootstrap_rounds": self.config.get("bootstrap_rounds", 10000),
                    "statistical_test": "NLPStatTest Bootstrap",
                },
                "statistics": all_stats,
                "comparisons": comparisons,
                "error_summary": error_stats,
            }
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"JSON detalhado salvo em {json_path}")

        # === CSV DE MÉTRICAS ===
        csv_path = os.path.join(self.output_dir, "metricas_resumo.csv")
        metrics_data = []
        for model_key, stats_dict in all_stats.items():
            row = {"Modelo": model_key}
            for metric in ["pos_accuracy", "lemma_accuracy", "uas", "las", "token_f1", "precision", "recall", "f1"]:
                if metric in stats_dict:
                    row[f"{metric}_mean"] = stats_dict[metric]["mean"]
                    row[f"{metric}_std"] = stats_dict[metric]["std"]
                    row[f"{metric}_ci_lower"] = stats_dict[metric].get("ci_lower", "")
                    row[f"{metric}_ci_upper"] = stats_dict[metric].get("ci_upper", "")
            metrics_data.append(row)

        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"CSV de métricas salvo em {csv_path}")

    def _empty_results(self) -> dict:
        return {
            "fold_scores": {
                "pos_accuracy": [],
                "lemma_accuracy": [],
                "uas": [],
                "las": [],
                "token_precision": [],
                "token_recall": [],
                "token_f1": [],
            },
        }

    def _empty_ner_results(self) -> dict:
        return {"precision": 0, "recall": 0, "f1": 0, "per_class": {}}


def main():
    """Função principal."""

    # Carregar config
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError("config.json não encontrado")

    # Inicializar avaliador
    evaluator = AcademicNLPEvaluator(config)

    # Carregar dados
    logger.info("Carregando corpus...")
    loader = DataLoader(
        conllu_path=config["data_path_conllu"],
        wikiner_path=config["data_path_ner"],
    )

    sentences = loader.load_conllu()
    wikiner_samples = loader.load_wikiner(max_sentences=config.get("ner_max_samples", 1000))
    logger.info(f"{len(sentences)} sentenças CONLL-U, {len(wikiner_samples)} WikiNER")

    all_results = {}
    all_stats = {}

    # === K-FOLD PARA SPACY, STANZA, UDPIPE ===
    for fw in ["spacy", "stanza", "udpipe"]:
        try:
            model_name = config["models"][fw]["name"]
            key = f"{fw}_{model_name}"
            logger.info(f"\n{'='*60}\nAvaliando {key}\n{'='*60}")

            result = evaluator.run_kfold_with_statistics(sentences, fw, model_name)
            all_results[key] = result

            # Compute statistics
            fold_scores = result["fold_scores"]
            stats = evaluator.compute_statistics(fold_scores)
            all_stats[key] = stats

            # Print resumo
            if "pos_accuracy" in stats:
                logger.info(f"  POS: {stats['pos_accuracy']['mean']:.4f} ± {stats['pos_accuracy']['std']:.4f}")
            if "uas" in stats:
                logger.info(f"  UAS: {stats['uas']['mean']:.4f} ± {stats['uas']['std']:.4f}")

        except Exception as e:
            logger.error(f"Erro em {fw}: {e}")
            continue

    # === NER COM SPACY ===
    try:
        spacy_name = config["models"]["spacy"]["name"]
        key = f"spacy_ner_{spacy_name}"
        logger.info(f"\n{'='*60}\nAvaliando {key}\n{'='*60}")

        ner_metrics = evaluator.run_ner_evaluation(wikiner_samples, spacy_name)
        all_results[key] = {"ner": ner_metrics}

        all_stats[key] = {
            "precision": {"mean": ner_metrics.get("precision", 0), "std": 0, "scores": [ner_metrics.get("precision", 0)]},
            "recall": {"mean": ner_metrics.get("recall", 0), "std": 0, "scores": [ner_metrics.get("recall", 0)]},
            "f1": {"mean": ner_metrics.get("f1", 0), "std": 0, "scores": [ner_metrics.get("f1", 0)]},
        }

        logger.info(f"  NER Precision: {ner_metrics.get('precision', 0):.4f}")
        logger.info(f"  NER Recall: {ner_metrics.get('recall', 0):.4f}")
        logger.info(f"  NER F1: {ner_metrics.get('f1', 0):.4f}")

        # Mostrar métricas por classe
        if "per_class" in ner_metrics:
            logger.info("\n  Métricas por classe:")
            for entity_class, class_metrics in ner_metrics["per_class"].items():
                logger.info(
                    f"    {entity_class}: "
                    f"P={class_metrics['precision']:.3f} "
                    f"R={class_metrics['recall']:.3f} "
                    f"F1={class_metrics['f1']:.3f} "
                    f"(n={class_metrics['support']})"
                )

    except Exception as e:
        logger.error(f"Erro em NER: {e}")

    # ========================================================================
    # TESTES ESTATÍSTICOS COM NLPStatTest
    # ========================================================================
    logger.info(f"\n{'='*60}\nTestes Estatísticos (NLPStatTest Bootstrap)\n{'='*60}")
    bootstrap_rounds = config.get("bootstrap_rounds", 10000)

    # Executar comparações com bootstrap
    comparisons = evaluator.run_bootstrap_comparisons(
        all_stats=all_stats,
        metrics_to_compare=["pos_accuracy", "uas", "las", "lemma_accuracy"],
        bootstrap_rounds=bootstrap_rounds
    )

    # === GERAR RELATÓRIOS ===
    evaluator.generate_academic_report(all_results, all_stats, comparisons)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Experimento concluído com sucesso!")
    logger.info(f"Arquivos salvos em: {evaluator.output_dir}/")
    logger.info("  - relatorio_academico.txt")
    logger.info("  - metricas_completas.json")
    logger.info("  - metricas_resumo.csv")
    logger.info("  - analise_erros_detalhada.txt")
    logger.info("  - pos_errors.csv")
    logger.info("  - lemma_errors.csv")
    logger.info("  - dependency_errors.csv")
    logger.info("  - ner_errors.csv")
    logger.info("  - estatisticas_erros.json")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
