# -*- coding: utf-8 -*-
"""
Integração de Bootstrap com NLPStatTest para comparação de NLP systems.

Referência: NLPStatTest Toolkit (Dror et al., 2018)
Paper: https://aclanthology.org/2020.aacl-demo.7.pdf

Implementa:
  - Bootstrap resampling com significância estatística
  - Teste de permutação para validação robusta
  - Cálculo de intervalo de confiança (95%)
  - P-values bilateral e unilateral
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger("NLPStatTest")

@dataclass
class BootstrapResult:
    """Resultado de teste estatístico bootstrap."""
    system_a: str
    system_b: str
    metric: str
    observed_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_bootstrap_rounds: int

    def __str__(self):
        sig_marker = "✓" if self.significant else "✗"
        return (
            f"{sig_marker} {self.system_a} vs {self.system_b} ({self.metric})\n"
            f"   Δμ = {self.observed_diff:+.6f}\n"
            f"   p-value = {self.p_value:.4f}\n"
            f"   IC 95% = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]"
        )


class BootstrapSignificanceTest:
    """
    Implementação de Bootstrap para testes estatísticos em NLP.
    
    Baseado em:
    - Dror et al. (2018): NLPStatTest - A Toolkit for Comparing NLP System Performance
    - Efron & Tibshirani (1993): Bootstrap methods
    """

    @staticmethod
    def resample_from_distribution(
        scores: np.ndarray,
        n_rounds: int = 10000,
        seed: int = 42
    ) -> np.ndarray:
        """
        Reamostragem com reposição (bootstrap resampling).
        
        Args:
            scores: Array 1D com scores por instância
            n_rounds: Número de resamples
            seed: Random seed
            
        Returns:
            Array com médias de cada resample
        """
        rng = np.random.default_rng(seed)
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_rounds):
            # Resample com reposição
            resample = rng.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))
        
        return np.array(bootstrap_means)

    @staticmethod
    def approximate_randomization_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_permutations: int = 10000,
        seed: int = 42
    ) -> float:
        """
        Approximate Randomization Test (permutation test).
        Mais robusto que bootstrap para dados pequenos.
        
        Args:
            scores_a: Scores do sistema A
            scores_b: Scores do sistema B
            n_permutations: Número de permutações
            seed: Random seed
            
        Returns:
            P-value bilateral
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Sistemas devem ter mesmo número de instâncias")
        
        rng = np.random.default_rng(seed)
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        count_extreme = 0
        combined = np.concatenate([scores_a, scores_b])
        n = len(scores_a)
        
        for _ in range(n_permutations):
            # Permuta os scores
            perm_idx = rng.permutation(len(combined))
            perm_a = combined[perm_idx[:n]]
            perm_b = combined[perm_idx[n:]]
            
            perm_diff = np.mean(perm_a) - np.mean(perm_b)
            
            # Conta diferenças >= observada (em valor absoluto)
            if np.abs(perm_diff) >= np.abs(observed_diff):
                count_extreme += 1
        
        p_value = count_extreme / n_permutations
        return p_value

    @staticmethod
    def bootstrap_hypothesis_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap_rounds: int = 10000,
        alpha: float = 0.05,
        seed: int = 42,
        test_type: str = "two_tailed"
    ) -> BootstrapResult:
        """
        Teste de hipótese via Bootstrap (NLPStatTest method).
        
        Hipótese nula: μ_a = μ_b (os sistemas não diferem)
        
        Args:
            scores_a: Array com scores do sistema A (por instância)
            scores_b: Array com scores do sistema B (por instância)
            n_bootstrap_rounds: Iterações de bootstrap
            alpha: Nível de significância (default 0.05)
            seed: Random seed
            test_type: "two_tailed", "greater", ou "less"
            
        Returns:
            BootstrapResult com estatísticas
        """
        if len(scores_a) != len(scores_b):
            raise ValueError(f"Tamanho diferente: {len(scores_a)} vs {len(scores_b)}")
        
        rng = np.random.default_rng(seed)
        n = len(scores_a)
        
        # Diferença observada
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        # Distribuição nula: H0 assume diferença = 0
        # Centra os dados em torno de 0
        pooled_mean = np.mean(np.concatenate([scores_a, scores_b]))
        centered_a = scores_a - np.mean(scores_a) + pooled_mean
        centered_b = scores_b - np.mean(scores_b) + pooled_mean
        combined = np.concatenate([centered_a, centered_b])
        
        # Bootstrap sob H0
        bootstrap_diffs = []
        for _ in range(n_bootstrap_rounds):
            idx = rng.choice(len(combined), size=n, replace=True)
            boot_a = combined[idx]
            boot_b = combined[np.setdiff1d(np.arange(len(combined)), idx)]
            
            if len(boot_b) == 0:
                boot_b = rng.choice(combined, size=n, replace=True)
            else:
                # Padding se necessário
                if len(boot_b) < n:
                    extra = rng.choice(combined, size=n - len(boot_b), replace=True)
                    boot_b = np.concatenate([boot_b, extra])
                elif len(boot_b) > n:
                    boot_b = boot_b[:n]
            
            bootstrap_diffs.append(np.mean(boot_a) - np.mean(boot_b))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Cálculo de p-value
        if test_type == "two_tailed":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif test_type == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        elif test_type == "less":
            p_value = np.mean(bootstrap_diffs <= observed_diff)
        else:
            raise ValueError(f"test_type desconhecido: {test_type}")
        
        # Intervalo de confiança (95%)
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        significant = p_value < alpha
        
        return BootstrapResult(
            system_a="System A",
            system_b="System B",
            metric="metric",
            observed_diff=float(observed_diff),
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            significant=significant,
            n_bootstrap_rounds=n_bootstrap_rounds
        )

    @staticmethod
    def compare_multiple_systems(
        systems_scores: Dict[str, np.ndarray],
        metric_name: str = "metric",
        n_bootstrap_rounds: int = 10000,
        alpha: float = 0.05,
        seed: int = 42
    ) -> Dict[str, List[BootstrapResult]]:
        """
        Comparação pairwise entre múltiplos sistemas.
        
        Args:
            systems_scores: Dict {system_name: np.array de scores}
            metric_name: Nome da métrica
            n_bootstrap_rounds: Iterações bootstrap
            alpha: Nível de significância
            seed: Random seed
            
        Returns:
            Dict com resultados para cada par
        """
        system_names = list(systems_scores.keys())
        results = {}
        
        for i, sys_a in enumerate(system_names):
            for sys_b in system_names[i+1:]:
                scores_a = systems_scores[sys_a]
                scores_b = systems_scores[sys_b]
                
                try:
                    result = BootstrapSignificanceTest.bootstrap_hypothesis_test(
                        scores_a,
                        scores_b,
                        n_bootstrap_rounds=n_bootstrap_rounds,
                        alpha=alpha,
                        seed=seed
                    )
                    
                    # Atualizar nomes
                    result.system_a = sys_a
                    result.system_b = sys_b
                    result.metric = metric_name
                    
                    key = f"{sys_a} vs {sys_b}"
                    results[key] = result
                    
                except Exception as e:
                    logger.warning(f"Erro ao comparar {sys_a} vs {sys_b}: {e}")
                    continue
        
        return results

    @staticmethod
    def paired_bootstrap_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap_rounds: int = 10000,
        seed: int = 42
    ) -> Dict:
        """
        Teste bootstrap pareado (quando A e B são dados aos mesmos exemplos).
        
        Args:
            scores_a: Scores pareados do sistema A
            scores_b: Scores pareados do sistema B
            n_bootstrap_rounds: Iterações
            seed: Random seed
            
        Returns:
            Dict com estatísticas
        """
        rng = np.random.default_rng(seed)
        
        # Diferenças pareadas
        diffs = scores_a - scores_b
        observed_diff = np.mean(diffs)
        
        # Bootstrap das diferenças
        bootstrap_means = []
        for _ in range(n_bootstrap_rounds):
            resample = rng.choice(diffs, size=len(diffs), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # P-value: quantos resamples têm diferença >= observada
        p_value = np.mean(np.abs(bootstrap_means) >= np.abs(observed_diff))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        significant = p_value < 0.05
        
        return {
            "observed_mean_diff": float(observed_diff),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant": significant,
            "bootstrap_means": bootstrap_means
        }


def format_results_table(
    results: Dict[str, BootstrapResult],
    alpha: float = 0.05
) -> str:
    """
    Formata resultados em tabela legível.
    """
    lines = [
        "=" * 100,
        f"{'Comparação':<40} {'Δμ':>12} {'P-value':>12} {'IC 95%':<20} {'Sig.':>8}",
        "-" * 100
    ]
    
    for comp_name, result in results.items():
        sig_str = "✓" if result.significant else ""
        ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        lines.append(
            f"{comp_name:<40} {result.observed_diff:>+12.6f} "
            f"{result.p_value:>12.4f} {ci_str:<20} {sig_str:>8}"
        )
    
    lines.append("=" * 100)
    return "\n".join(lines)


if __name__ == "__main__":
    # Exemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Simulando 3 sistemas com scores em 100 instâncias
    np.random.seed(42)
    scores_spacy = np.random.normal(0.97, 0.01, 100)
    scores_stanza = np.random.normal(0.96, 0.015, 100)
    scores_udpipe = np.random.normal(0.968, 0.012, 100)
    
    systems = {
        "spaCy": scores_spacy,
        "Stanza": scores_stanza,
        "UDPipe": scores_udpipe
    }
    
    # Executar testes
    results = BootstrapSignificanceTest.compare_multiple_systems(
        systems,
        metric_name="POS Accuracy",
        n_bootstrap_rounds=10000,
        seed=42
    )
    
    print(format_results_table(results))
    for name, result in results.items():
        print(f"\n{result}")