# -*- coding: utf-8 -*-
"""
Visualizações dos Resultados de Avaliação PLN.

Gera:
  1. Gráfico Comparativo (spaCy vs Stanza vs UDPipe) - por métrica
  2. Distribuição dos Scores por Fold (boxplot + scatter)
  3. Heatmap de P-values (teste de significância)

Uso:
  python plot_results.py

Requer:
  pip install matplotlib seaborn numpy pandas
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ResultsVisualizer:
    """Cria visualizações dos resultados de avaliação PLN."""
    
    def __init__(self, json_path: str = "results/metricas_completas.json", output_dir: str = "results/plots"):
        """
        Inicializa visualizador.
        
        Args:
            json_path: Caminho do JSON com resultados
            output_dir: Diretório para salvar gráficos
        """
        self.json_path = json_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar dados
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.statistics = self.data.get('statistics', {})
        self.comparisons = self.data.get('comparisons', {})
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 2
    
    # ========================================================================
    # GRÁFICO 1: COMPARATIVO POR MÉTRICA
    # ========================================================================
    
    def plot_comparative_metrics(self):
        """
        Gráfico comparativo com barras para cada métrica.
        Mostra: média, desvio padrão e intervalo de confiança.
        """
        metrics_to_plot = ["pos_accuracy", "lemma_accuracy", "uas", "las"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        colors = {
            "spacy_pt_core_news_lg": "#FF6B6B",
            "stanza_pt": "#4ECDC4",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe": "#45B7D1"
        }
        
        labels_map = {
            "spacy_pt_core_news_lg": "spaCy",
            "stanza_pt": "Stanza",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe": "UDPipe"
        }
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            systems = []
            means = []
            stds = []
            cis_lower = []
            cis_upper = []
            
            for system, stats_dict in self.statistics.items():
                if metric in stats_dict and system != "huggingface_ner_pierreguillou/bert-base-cased-pt-lenerbr":
                    systems.append(labels_map.get(system, system))
                    means.append(stats_dict[metric]["mean"])
                    stds.append(stats_dict[metric]["std"])
                    cis_lower.append(stats_dict[metric]["ci_lower"])
                    cis_upper.append(stats_dict[metric]["ci_upper"])
            
            x_pos = np.arange(len(systems))
            errors = [
                np.array(means) - np.array(cis_lower),
                np.array(cis_upper) - np.array(means)
            ]
            
            bars = ax.bar(x_pos, means, yerr=errors, capsize=10, alpha=0.7, 
                         color=[colors[list(self.statistics.keys())[i]] for i in range(len(systems))])
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(systems, fontsize=11)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + 0.02, f"{m:.3f}\n±{s:.3f}", ha='center', va='bottom', fontsize=9)
        
        plt.suptitle("Comparação de Métricas (Média ± IC 95%)", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / "01_metricas_comparativas.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_path}")
        plt.close()
    
    # ========================================================================
    # GRÁFICO 2: DISTRIBUIÇÃO POR FOLD (BOXPLOT + SCATTER)
    # ========================================================================
    
    def plot_fold_distribution(self):
        """
        Gráfico com distribuição dos scores por fold.
        Mostra boxplot + pontos individuais por fold.
        """
        metrics_to_plot = ["pos_accuracy", "lemma_accuracy", "uas", "las"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        labels_map = {
            "spacy_pt_core_news_lg": "spaCy",
            "stanza_pt": "Stanza",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe": "UDPipe"
        }
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            data_for_box = []
            labels_for_box = []
            
            for system, stats_dict in self.statistics.items():
                if metric in stats_dict and system != "huggingface_ner_pierreguillou/bert-base-cased-pt-lenerbr":
                    scores = stats_dict[metric]["scores"]
                    data_for_box.append(scores)
                    labels_for_box.append(labels_map.get(system, system))
            
            # Boxplot
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                           notch=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            
            # Colorir caixas
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Adicionar pontos individuais (scatter)
            for i, (scores, label) in enumerate(zip(data_for_box, labels_for_box)):
                x_pos = np.random.normal(i+1, 0.04, size=len(scores))
                ax.scatter(x_pos, scores, alpha=0.4, s=50, color='black', zorder=3)
            
            ax.set_ylabel("Score", fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()} - Distribuição por Fold", fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle("Distribuição dos Scores por Fold (Boxplot + Pontos)", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / "02_distribuicao_folds.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_path}")
        plt.close()
    
    # ========================================================================
    # GRÁFICO 3: HEATMAP DE P-VALUES
    # ========================================================================
    
    def plot_pvalue_heatmap(self):
        """
        Heatmap com p-values das comparações estatísticas.
        Verde = significativo (p < 0.05), Vermelho = não significativo.
        """
        metrics = ["pos_accuracy", "uas", "las", "lemma_accuracy"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        systems_list = [
            "spacy_pt_core_news_lg",
            "stanza_pt",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe"
        ]
        
        labels_map = {
            "spacy_pt_core_news_lg": "spaCy",
            "stanza_pt": "Stanza",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe": "UDPipe"
        }
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric not in self.comparisons:
                continue
            
            # Criar matriz de p-values
            matrix = np.ones((3, 3))
            np.fill_diagonal(matrix, 1.0)  # Diagonal = 1 (sem comparação consigo mesmo)
            
            metric_comparisons = self.comparisons[metric]
            
            for i, sys_a in enumerate(systems_list):
                for j, sys_b in enumerate(systems_list):
                    if i >= j:
                        continue
                    
                    comp_name = f"{sys_a} vs {sys_b}"
                    if comp_name in metric_comparisons:
                        p_val = metric_comparisons[comp_name].get("p_value", 1.0)
                        matrix[i, j] = p_val
                        matrix[j, i] = p_val
            
            # Criar heatmap
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn_r',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'P-value'},
                xticklabels=[labels_map.get(s, s) for s in systems_list],
                yticklabels=[labels_map.get(s, s) for s in systems_list],
                ax=ax,
                square=True,
                linewidths=1,
                linecolor='black'
            )
            
            ax.set_title(f"{metric.replace('_', ' ').title()}\n(Verde=sig. p<0.05, Vermelho=não sig.)", 
                        fontsize=11, fontweight='bold')
        
        plt.suptitle("Heatmap de P-values (NLPStatTest Bootstrap)", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / "03_heatmap_pvalues.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_path}")
        plt.close()
    
    # ========================================================================
    # GRÁFICO 4: LINHA DO TEMPO (SCORES POR FOLD)
    # ========================================================================
    
    def plot_fold_timeseries(self):
        """
        Gráfico de linha mostrando evolução dos scores ao longo dos folds.
        """
        metrics_to_plot = ["pos_accuracy", "lemma_accuracy", "uas", "las"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        labels_map = {
            "spacy_pt_core_news_lg": "spaCy",
            "stanza_pt": "Stanza",
            "udpipe_models/portuguese-bosque-ud-2.5-191206.udpipe": "UDPipe"
        }
        
        colors_dict = {
            "spaCy": "#FF6B6B",
            "Stanza": "#4ECDC4",
            "UDPipe": "#45B7D1"
        }
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            for system, stats_dict in self.statistics.items():
                if metric in stats_dict and system != "huggingface_ner_pierreguillou/bert-base-cased-pt-lenerbr":
                    scores = stats_dict[metric]["scores"]
                    folds = np.arange(1, len(scores) + 1)
                    label = labels_map.get(system, system)
                    
                    ax.plot(folds, scores, marker='o', linewidth=2.5, markersize=8,
                           label=label, color=colors_dict[label], alpha=0.8)
                    
                    # Adicionar valor da média como linha tracejada
                    mean_val = np.mean(scores)
                    ax.axhline(y=mean_val, linestyle='--', alpha=0.3, color=colors_dict[label])
            
            ax.set_xlabel("Fold", fontsize=11)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, 6))
            ax.set_ylim([0, 1.0])
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Evolução dos Scores ao Longo dos Folds", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / "04_scores_por_fold.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_path}")
        plt.close()
    
    # ========================================================================
    # GRÁFICO 5: DIFERENÇAS MÉDIAS (DELTA-MU COM IC)
    # ========================================================================
    
    def plot_mean_differences(self):
        """
        Gráfico de diferenças médias entre sistemas com IC 95%.
        """
        metrics_to_plot = ["pos_accuracy", "uas", "las", "lemma_accuracy"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in self.comparisons:
                continue
            
            comparisons_data = self.comparisons[metric]
            comp_names = []
            deltas = []
            ci_lower_list = []
            ci_upper_list = []
            p_values = []
            
            for comp_name, result in comparisons_data.items():
                comp_names.append(comp_name)
                deltas.append(result.get("mean_diff", 0))
                ci_lower_list.append(result.get("ci_lower", 0))
                ci_upper_list.append(result.get("ci_upper", 0))
                p_values.append(result.get("p_value", 1.0))
            
            # Calcular erros
            errors = [
                np.array(deltas) - np.array(ci_lower_list),
                np.array(ci_upper_list) - np.array(deltas)
            ]
            
            # Cores baseadas em significância
            colors = ['#2ECC71' if p < 0.05 else '#E74C3C' for p in p_values]
            
            x_pos = np.arange(len(comp_names))
            ax.barh(x_pos, deltas, xerr=errors, capsize=8, alpha=0.7, color=colors)
            
            # Linha vertical em zero (sem diferença)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels([c.replace(" vs ", "\nvs ") for c in comp_names], fontsize=9)
            ax.set_xlabel("Δμ (Diferença Média)", fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Legenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ECC71', alpha=0.7, label='Significativo (p<0.05)'),
                Patch(facecolor='#E74C3C', alpha=0.7, label='Não significativo (p≥0.05)')
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        plt.suptitle("Diferenças Médias entre Sistemas (Δμ ± IC 95%)", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / "05_diferencas_medias.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_path}")
        plt.close()
    
    # ========================================================================
    # EXECUTAR TUDO
    # ========================================================================
    
    def generate_all_plots(self):
        """Gera todos os gráficos."""
        print("\n" + "="*60)
        print("GERANDO GRÁFICOS DE RESULTADOS PLN")
        print("="*60 + "\n")
        
        print("1/5 - Gráfico Comparativo de Métricas...")
        self.plot_comparative_metrics()
        
        print("2/5 - Distribuição dos Scores por Fold...")
        self.plot_fold_distribution()
        
        print("3/5 - Heatmap de P-values...")
        self.plot_pvalue_heatmap()
        
        print("4/5 - Evolução dos Scores por Fold...")
        self.plot_fold_timeseries()
        
        print("5/5 - Diferenças Médias com IC 95%...")
        self.plot_mean_differences()
        
        print("\n" + "="*60)
        print(f"✓ Todos os gráficos salvos em: {self.output_dir}/")
        print("="*60 + "\n")
        
        # Listar arquivos gerados
        plots = list(self.output_dir.glob("*.png"))
        for i, plot in enumerate(sorted(plots), 1):
            print(f"  {i}. {plot.name}")


def main():
    """Função principal."""
    
    # Usar caminhos padrão
    json_path = "results/metricas_completas.json"
    output_dir = "results/plots"
    
    try:
        visualizer = ResultsVisualizer(json_path=json_path, output_dir=output_dir)
        visualizer.generate_all_plots()
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        print(f"Certifique-se de que {json_path} existe e execute main.py primeiro.")
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")


if __name__ == "__main__":
    main()