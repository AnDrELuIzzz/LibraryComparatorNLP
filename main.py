# -*- coding: utf-8 -*-
"""
Sistema Completo de Avaliação Comparativa de Ferramentas PLN para Português Brasileiro.

Implementa:
  - K-Fold Cross-Validation
  - Bootstrap com NLPStatTest (Dror et al., 2018)
  - Testes de Permutação (Approximate Randomization Test)
  - Relatórios científicos (TXT, JSON, CSV, LaTeX)
  
Uso:
  python main.py

Referências:
  - Dror et al. (2018): NLPStatTest - A Toolkit for Comparing NLP System Performance
  - Efron & Tibshirani (1993): Bootstrap Methods
"""

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
        
        error_analysis = {
            "pos_errors": defaultdict(lambda: {"gold": [], "pred": []}),
            "lemma_errors": [],
            "dep_errors": [],
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
            "error_analysis": error_analysis,
        }
    
    def run_ner_evaluation(self, wikiner_samples: list, model_name: str) -> dict:
        """Avalia NER."""
        logger.info(f"[NER] Avaliando HuggingFace ({model_name})")
        
        try:
            wrapper = get_model_wrapper("huggingface_ner", model_name)
            wrapper.load_model()
        except Exception as e:
            logger.error(f"Erro ao carregar NER: {e}")
            return self._empty_ner_results()
        
        all_gold, all_pred = [], []
        ner_by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for s in tqdm(wikiner_samples, desc="NER", leave=False):
            tokens = s["tokens"]
            gold = s["tags"]
            
            try:
                pred = wrapper.ner(tokens)
            except Exception as e:
                logger.warning(f"Erro em NER: {e}")
                continue
            
            length = min(len(gold), len(pred))
            all_gold.extend(gold[:length])
            all_pred.extend(pred[:length])
            
            # Por classe
            for g, p in zip(gold[:length], pred[:length]):
                entity_class = g.split("-")[-1] if "-" in g else g
                
                if g == p and g != "O":
                    ner_by_class[entity_class]["tp"] += 1
                elif g != "O" and p == "O":
                    ner_by_class[entity_class]["fn"] += 1
                elif g == "O" and p != "O":
                    ner_by_class[p.split("-")[-1]]["fp"] += 1
        
        metrics = EvaluationMetrics.compute_ner_metrics(all_gold, all_pred)
        
        # Por classe
        metrics["per_class"] = {}
        for entity_class, counts in ner_by_class.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            metrics["per_class"][entity_class] = {"precision": prec, "recall": rec, "f1": f1}
        
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
    # NOVOS MÉTODOS COM NLPStatTest
    # ========================================================================
    
    def run_bootstrap_comparisons(
        self,
        all_stats: dict,
        metrics_to_compare: list = None,
        bootstrap_rounds: int = 10000
    ) -> dict:
        """
        Executa comparações pairwise com bootstrap automático (NLPStatTest).
        
        Args:
            all_stats: Dicionário com estatísticas de todos os modelos
            metrics_to_compare: Lista de métricas a comparar
            bootstrap_rounds: Iterações de bootstrap
            
        Returns:
            Dict com resultados organizados por métrica
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
    
    def run_permutation_test_comparisons(
        self,
        all_stats: dict,
        metrics_to_compare: list = None,
        n_permutations: int = 10000
    ) -> dict:
        """
        Executa comparações com Approximate Randomization Test (mais robusto).
        
        Args:
            all_stats: Dicionário com estatísticas
            metrics_to_compare: Métricas a comparar
            n_permutations: Número de permutações
            
        Returns:
            Dict com resultados do teste de permutação
        """
        if metrics_to_compare is None:
            metrics_to_compare = ["pos_accuracy", "uas", "las"]
        
        comparisons = {}
        model_keys = list(all_stats.keys())
        
        for metric in metrics_to_compare:
            logger.info(f"\n[Permutation Test] Testando {metric}...")
            comparisons[metric] = {}
            
            for i, key_a in enumerate(model_keys):
                for key_b in model_keys[i+1:]:
                    if metric not in all_stats[key_a] or metric not in all_stats[key_b]:
                        continue
                    
                    scores_a = np.array(all_stats[key_a][metric]["scores"])
                    scores_b = np.array(all_stats[key_b][metric]["scores"])
                    
                    p_value = BootstrapSignificanceTest.approximate_randomization_test(
                        scores_a,
                        scores_b,
                        n_permutations=n_permutations,
                        seed=self.config["seed"]
                    )
                    
                    comp_name = f"{key_a} vs {key_b}"
                    comparisons[metric][comp_name] = {
                        "p_value": p_value,
                        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
                        "significant": p_value < 0.05,
                        "test_type": "approximate_randomization",
                        "n_permutations": n_permutations,
                    }
                    
                    sig_marker = "✓" if p_value < 0.05 else "✗"
                    logger.info(f"  {sig_marker} {comp_name}: p={p_value:.4f}")
        
        return comparisons
    
    def run_paired_bootstrap_tests(
        self,
        all_stats: dict,
        metrics_to_compare: list = None,
        bootstrap_rounds: int = 10000
    ) -> dict:
        """
        Usa bootstrap pareado (K-Fold sincronizados).
        
        Args:
            all_stats: Dicionário com estatísticas
            metrics_to_compare: Métricas a comparar
            bootstrap_rounds: Iterações de bootstrap
            
        Returns:
            Dict com resultados de teste pareado
        """
        if metrics_to_compare is None:
            metrics_to_compare = ["pos_accuracy", "uas", "las"]
        
        comparisons = {}
        model_keys = list(all_stats.keys())
        
        for metric in metrics_to_compare:
            logger.info(f"\n[Paired Bootstrap] Testando {metric}...")
            comparisons[metric] = {}
            
            for i, key_a in enumerate(model_keys):
                for key_b in model_keys[i+1:]:
                    if metric not in all_stats[key_a] or metric not in all_stats[key_b]:
                        continue
                    
                    scores_a = np.array(all_stats[key_a][metric]["scores"])
                    scores_b = np.array(all_stats[key_b][metric]["scores"])
                    
                    if len(scores_a) != len(scores_b):
                        logger.warning(
                            f"Tamanhos diferentes: {key_a} ({len(scores_a)}) vs "
                            f"{key_b} ({len(scores_b)})"
                        )
                        continue
                    
                    result = BootstrapSignificanceTest.paired_bootstrap_test(
                        scores_a,
                        scores_b,
                        n_bootstrap_rounds=bootstrap_rounds,
                        seed=self.config["seed"]
                    )
                    
                    comp_name = f"{key_a} vs {key_b}"
                    comparisons[metric][comp_name] = result
                    comparisons[metric][comp_name]["test_type"] = "paired_bootstrap"
                    
                    sig_marker = "✓" if result["significant"] else "✗"
                    logger.info(f"  {sig_marker} {comp_name}: p={result['p_value']:.4f}")
        
        return comparisons
    
    # ========================================================================
    # MÉTODO LEGADO (mantido para compatibilidade)
    # ========================================================================
    
    def statistical_test_bootstrap(self, scores_a: np.ndarray, scores_b: np.ndarray, rounds: int = 10000) -> dict:
        """Bootstrap test simples (legado - use NLPStatTest para novos trabalhos)."""
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"p_value": None, "mean_diff": None}
        
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        combined = np.concatenate([scores_a, scores_b])
        diffs = []
        
        for _ in range(rounds):
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            sample_a = resampled[:len(scores_a)]
            sample_b = resampled[len(scores_a):]
            diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        p_value = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / rounds
        
        return {
            "p_value": float(p_value),
            "mean_diff": float(observed_diff),
            "ci_lower": float(np.percentile(diffs, 2.5)),
            "ci_upper": float(np.percentile(diffs, 97.5)),
        }
    
    def generate_academic_report(self, all_results: dict, all_stats: dict, comparisons: dict):
        """Gera relatório científico em TXT, JSON, CSV e LaTeX."""
        
        # === RELATÓRIO TXT ===
        txt_path = os.path.join(self.output_dir, "relatorio_academico.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write("RELATÓRIO COMPLETO - AVALIAÇÃO COMPARATIVA DE FERRAMENTAS DE PLN\n")
            f.write("(Com Testes Estatísticos via NLPStatTest Bootstrap)\n")
            f.write("=" * 120 + "\n\n")
            
            # Metadados
            f.write("METADADOS\n")
            f.write("-" * 120 + "\n")
            f.write(f"Data/Hora: {self.timestamp}\n")
            f.write(f"K-Folds: {self.config['n_splits']}\n")
            f.write(f"Bootstrap Rounds (NLPStatTest): {self.config.get('bootstrap_rounds', 10000)}\n")
            f.write(f"Random Seed: {self.config['seed']}\n")
            f.write(f"Nível de Significância (α): 0.05\n\n")
            
            # Tabela de resumo
            f.write("RESUMO DE RESULTADOS (Média ± Desvio Padrão)\n")
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
                        row.append(f"{mean:.4f} ± {std:.4f}")
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
                f.write("MÉTRICAS DE NER (Precisão, Revocação, F1)\n")
                f.write("-" * 120 + "\n")
                ner_headers = ["Modelo", "Precisão", "Revocação", "F1"]
                f.write(tabulate(ner_rows, headers=ner_headers, tablefmt="grid") + "\n\n")
            
            # Testes estatísticos
            f.write("TESTES ESTATÍSTICOS (NLPStatTest Bootstrap, p < 0.05 = significativo)\n")
            f.write("-" * 120 + "\n")
            
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
            
            f.write("\n" + "=" * 120 + "\n")
            f.write("INTERPRETAÇÃO:\n")
            f.write("-" * 120 + "\n")
            f.write(
                "✓ Significativo: A diferença entre sistemas é estatisticamente significativa (p < 0.05).\n"
                "✗ Não significativo: A diferença pode ser atribuída ao acaso (p >= 0.05).\n"
                "IC 95%: Intervalo de confiança de 95% (não contém zero → significativo).\n\n"
            )
            f.write("REFERÊNCIA:\n")
            f.write(
                "Dror, R., Baum, G., Marinov, M., Avramovic, M., Roth, A., Chokshi, A., & "
                "Dagan, I. (2018). NLPStatTest - A Toolkit for Comparing NLP System Performance. "
                "In Proceedings of the 2020 Conference on Asia-Pacific Chapter of the Association "
                "for Computational Linguistics.\n"
            )
            f.write("=" * 120 + "\n")
        
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
            }
            
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"JSON detalhado salvo em {json_path}")
        
        # === CSV ===
        csv_path = os.path.join(self.output_dir, "resultados_kfold.csv")
        
        csv_rows = []
        for model_key, stats_dict in all_stats.items():
            for metric, stat_info in stats_dict.items():
                if isinstance(stat_info, dict) and "scores" in stat_info:
                    for fold_id, score in enumerate(stat_info["scores"], 1):
                        csv_rows.append({
                            "modelo": model_key,
                            "metrica": metric,
                            "fold": fold_id,
                            "score": score,
                        })
        
        df_csv = pd.DataFrame(csv_rows)
        df_csv.to_csv(csv_path, index=False)
        
        logger.info(f"CSV salvo em {csv_path}")
        
        # === LaTeX ===
        latex_path = os.path.join(self.output_dir, "tabelas_latex.tex")
        
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("% Tabelas para inserir no artigo LaTeX\n")
            f.write("% Gerado automaticamente - Sistema de Avaliação PLN\n\n")
            
            # Tabela 1: Resumo
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Resultados Comparativos de Ferramentas PLN (Média ± Desvio Padrão)}\n")
            f.write("\\label{tab:resultados}\n")
            f.write("\\begin{tabular}{lccccccc}\n")
            f.write("\\toprule\n")
            f.write("Modelo & POS & Lemma & UAS & LAS & Tok Prec & Tok Rec & Tok F1 \\\\\n")
            f.write("\\midrule\n")
            
            for model_key, stats_dict in all_stats.items():
                model_short = model_key.replace("_", "\\_")
                parts = [model_short]
                
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
                        parts.append(f"${mean:.3f} \\pm {std:.3f}$")
                    else:
                        parts.append("N/A")
                
                f.write(" & ".join(parts) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

            # Tabela 2: NER
            ner_rows = [
                (
                    model_key.replace("_", "\\_"),
                    stats_dict["precision"]["mean"],
                    stats_dict["recall"]["mean"],
                    stats_dict["f1"]["mean"],
                )
                for model_key, stats_dict in all_stats.items()
                if all(metric in stats_dict for metric in ["precision", "recall", "f1"])
            ]

            if ner_rows:
                f.write("\n\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Resultados de NER (Precisão/Revocação/F1)}\n")
                f.write("\\label{tab:ner}\n")
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\toprule\n")
                f.write("Modelo & Precisão & Revocação & F1 \\\\\n")
                f.write("\\midrule\n")
                for name, prec, rec, f1_score in ner_rows:
                    f.write(f"{name} & {prec:.3f} & {rec:.3f} & {f1_score:.3f} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            # Tabela 3: Comparações Bootstrap
            f.write("\n% Tabela de Comparações Estatísticas (Bootstrap NLPStatTest)\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparações Estatísticas (NLPStatTest Bootstrap, p < 0.05 = significativo)}\n")
            f.write("\\label{tab:comparacoes}\n")
            f.write("\\begin{tabular}{llrrr}\n")
            f.write("\\toprule\n")
            f.write("Métrica & Comparação & $\\Delta\\mu$ & $p$-value & Sig. \\\\\n")
            f.write("\\midrule\n")
            
            for metric_name, metric_comparisons in comparisons.items():
                for i, (comp_name, result) in enumerate(metric_comparisons.items()):
                    if isinstance(result, dict) and result.get("p_value"):
                        p_val = result["p_value"]
                        mean_diff = result["mean_diff"]
                        sig = "\\checkmark" if p_val < 0.05 else ""
                        
                        metric_display = metric_name if i == 0 else ""
                        f.write(
                            f"{metric_display} & {comp_name} & {mean_diff:+.4f} & "
                            f"{p_val:.4f} & {sig} \\\\\n"
                        )
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"LaTeX salvo em {latex_path}")
    
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
            "error_analysis": {},
        }
    
    def _empty_ner_results(self) -> dict:
        return {"precision": 0, "recall": 0, "f1": 0, "per_class": {}}


def format_results_table_simple(metric_comparisons: dict) -> str:
    """Tabela formatada simples para logging."""
    lines = [
        f"{'Comparação':<50} {'Δμ':>10} {'P-value':>10} {'IC 95%':<25} {'Sig':>5}",
        "-" * 105
    ]
    
    for comp_name, result in metric_comparisons.items():
        if isinstance(result, dict) and result.get("p_value"):
            sig = "✓" if result["significant"] else ""
            ci_lower = result.get("ci_lower", 0)
            ci_upper = result.get("ci_upper", 0)
            ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            
            lines.append(
                f"{comp_name:<50} {result['mean_diff']:>+10.6f} "
                f"{result['p_value']:>10.4f} {ci_str:<25} {sig:>5}"
            )
    
    return "\n".join(lines)


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
    wikiner_samples = loader.load_wikiner(max_sentences=1000)
    
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
    
    # === NER COM HUGGINGFACE ===
    try:
        hf_name = config["models"]["huggingface"]["general"]["name"]
        key = f"huggingface_ner_{hf_name}"
        
        logger.info(f"\n{'='*60}\nAvaliando {key}\n{'='*60}")
        
        ner_metrics = evaluator.run_ner_evaluation(wikiner_samples, hf_name)
        all_results[key] = {"ner": ner_metrics}
        
        all_stats[key] = {
            "precision": {"mean": ner_metrics.get("precision", 0), "std": 0, "scores": [ner_metrics.get("precision", 0)]},
            "recall": {"mean": ner_metrics.get("recall", 0), "std": 0, "scores": [ner_metrics.get("recall", 0)]},
            "f1": {"mean": ner_metrics.get("f1", 0), "std": 0, "scores": [ner_metrics.get("f1", 0)]},
        }
        
        logger.info(f"  NER F1: {ner_metrics.get('f1', 0):.4f}")
    
    except Exception as e:
        logger.error(f"Erro em NER: {e}")
    
    # ========================================================================
    # TESTES ESTATÍSTICOS COM NLPStatTest (NOVO!)
    # ========================================================================
    
    logger.info(f"\n{'='*60}\nTestes Estatísticos (NLPStatTest Bootstrap)\n{'='*60}")
    
    bootstrap_rounds = config.get("bootstrap_rounds", 10000)
    
    # Executar comparações com bootstrap
    comparisons = evaluator.run_bootstrap_comparisons(
        all_stats=all_stats,
        metrics_to_compare=["pos_accuracy", "uas", "las", "lemma_accuracy"],
        bootstrap_rounds=bootstrap_rounds
    )
    
    # Exibir tabelas formatadas
    for metric, metric_comparisons in comparisons.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(format_results_table_simple(metric_comparisons))
    
    # (OPCIONAL) Executar teste de permutação como validação
    logger.info(f"\n{'='*60}\nValidação: Approximate Randomization Test\n{'='*60}")
    permutation_results = evaluator.run_permutation_test_comparisons(
        all_stats=all_stats,
        metrics_to_compare=["pos_accuracy", "uas", "las"],
        n_permutations=10000
    )
    
    # (OPCIONAL) Executar teste pareado
    logger.info(f"\n{'='*60}\nValidação: Paired Bootstrap Test\n{'='*60}")
    paired_results = evaluator.run_paired_bootstrap_tests(
        all_stats=all_stats,
        metrics_to_compare=["pos_accuracy", "uas", "las"],
        bootstrap_rounds=bootstrap_rounds
    )
    
    # === GERAR RELATÓRIOS ===
    evaluator.generate_academic_report(all_results, all_stats, comparisons)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Experimento concluído com sucesso!")
    logger.info(f"Arquivos salvos em: {evaluator.output_dir}/")
    logger.info("  - relatorio_academico.txt")
    logger.info("  - metricas_completas.json")
    logger.info("  - resultados_kfold.csv")
    logger.info("  - tabelas_latex.tex")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()