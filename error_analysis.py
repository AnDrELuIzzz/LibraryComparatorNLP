
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
import json
import os

class ErrorAnalyzer:
    """Análise detalhada de erros para artigo científico."""

    def __init__(self):
        self.pos_errors = []
        self.lemma_errors = []
        self.dep_errors = []
        self.ner_errors = []

    def collect_errors_from_prediction(
        self,
        sent_id: int,
        text: str,
        gold_tokens: List[str],
        gold_pos: List[str],
        gold_lemmas: List[str],
        gold_heads: List[int],
        gold_deprels: List[str],
        pred_pos: List[str],
        pred_lemmas: List[str],
        pred_heads: List[int],
        pred_deprels: List[str],
        model_name: str = "unknown"
    ):
        """Coleta erros token por token."""
        length = min(
            len(gold_tokens), 
            len(pred_pos), 
            len(pred_lemmas), 
            len(pred_heads), 
            len(pred_deprels)
        )

        for i in range(length):
            token = gold_tokens[i]

            # POS Errors
            if gold_pos[i] != pred_pos[i]:
                self.pos_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold_pos": gold_pos[i],
                    "predicted_pos": pred_pos[i],
                    "model": model_name,
                    "error_type": f"{gold_pos[i]}_to_{pred_pos[i]}"
                })

            # Lemma Errors
            if gold_lemmas[i] != pred_lemmas[i]:
                self.lemma_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold_lemma": gold_lemmas[i],
                    "predicted_lemma": pred_lemmas[i],
                    "pos_tag": gold_pos[i],
                    "model": model_name
                })

            # Dependency Errors
            if gold_heads[i] != pred_heads[i] or gold_deprels[i] != pred_deprels[i]:
                self.dep_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": i + 1,
                    "gold_head": gold_heads[i],
                    "gold_deprel": gold_deprels[i],
                    "predicted_head": pred_heads[i],
                    "predicted_deprel": pred_deprels[i],
                    "pos_tag": gold_pos[i],
                    "model": model_name,
                    "head_correct": gold_heads[i] == pred_heads[i],
                    "deprel_correct": gold_deprels[i] == pred_deprels[i]
                })

    def collect_ner_errors(
        self,
        tokens: List[str],
        gold_tags: List[str],
        pred_tags: List[str],
        model_name: str = "unknown",
        sent_id: int = 0
    ):
        """Coleta erros de NER."""
        text = " ".join(tokens)
        length = min(len(tokens), len(gold_tags), len(pred_tags))

        for i in range(length):
            if gold_tags[i] != pred_tags[i]:
                self.ner_errors.append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": tokens[i],
                    "position": i + 1,
                    "gold_tag": gold_tags[i],
                    "predicted_tag": pred_tags[i],
                    "model": model_name
                })

    def get_error_statistics(self) -> Dict:
        """Retorna estatísticas dos erros."""
        return {
            "pos_errors": {
                "total": len(self.pos_errors),
                "most_common": Counter([e["error_type"] for e in self.pos_errors]).most_common(10)
            },
            "lemma_errors": {
                "total": len(self.lemma_errors),
                "by_pos": Counter([e["pos_tag"] for e in self.lemma_errors]).most_common(10)
            },
            "dep_errors": {
                "total": len(self.dep_errors),
                "head_only_errors": sum(1 for e in self.dep_errors if not e["head_correct"] and e["deprel_correct"]),
                "deprel_only_errors": sum(1 for e in self.dep_errors if e["head_correct"] and not e["deprel_correct"]),
                "both_errors": sum(1 for e in self.dep_errors if not e["head_correct"] and not e["deprel_correct"])
            },
            "ner_errors": {
                "total": len(self.ner_errors),
                "most_common": Counter([f"{e['gold_tag']}_to_{e['predicted_tag']}" for e in self.ner_errors]).most_common(10)
            }
        }

    def save_to_csv(self, output_dir: str = "results"):
        """Salva erros em arquivos CSV separados."""
        os.makedirs(output_dir, exist_ok=True)

        # POS Errors
        if self.pos_errors:
            df_pos = pd.DataFrame(self.pos_errors)
            df_pos.to_csv(f"{output_dir}/pos_errors.csv", index=False, encoding="utf-8")

        # Lemma Errors
        if self.lemma_errors:
            df_lemma = pd.DataFrame(self.lemma_errors)
            df_lemma.to_csv(f"{output_dir}/lemma_errors.csv", index=False, encoding="utf-8")

        # Dependency Errors
        if self.dep_errors:
            df_dep = pd.DataFrame(self.dep_errors)
            df_dep.to_csv(f"{output_dir}/dependency_errors.csv", index=False, encoding="utf-8")

        # NER Errors
        if self.ner_errors:
            df_ner = pd.DataFrame(self.ner_errors)
            df_ner.to_csv(f"{output_dir}/ner_errors.csv", index=False, encoding="utf-8")

    def save_detailed_report(self, output_path: str):
        """Salva relatório detalhado de erros para análise acadêmica."""
        stats = self.get_error_statistics()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("RELATÓRIO DETALHADO DE ERROS - ANÁLISE PARA ARTIGO CIENTÍFICO\n")
            f.write("=" * 100 + "\n\n")

            # POS Errors
            f.write("\n" + "="*100 + "\n")
            f.write("1. ERROS DE POS TAGGING\n")
            f.write("="*100 + "\n")
            f.write(f"Total de erros: {stats['pos_errors']['total']}\n\n")

            if stats['pos_errors']['most_common']:
                f.write("Erros mais comuns (Gold → Predicted):\n")
                for error_type, count in stats['pos_errors']['most_common']:
                    f.write(f"  {error_type}: {count} ocorrências\n")

            f.write("\nExemplos de erros (primeiros 20):\n")
            f.write("-" * 100 + "\n")
            for i, error in enumerate(self.pos_errors[:20], 1):
                f.write(f"\nErro {i}:\n")
                f.write(f"  Sentença: {error['text']}\n")
                f.write(f"  Token: '{error['token']}' (posição {error['position']})\n")
                f.write(f"  Gold POS: {error['gold_pos']}\n")
                f.write(f"  Predicted POS: {error['predicted_pos']}\n")
                f.write(f"  Modelo: {error['model']}\n")

            # Lemma Errors
            f.write("\n\n" + "="*100 + "\n")
            f.write("2. ERROS DE LEMATIZAÇÃO\n")
            f.write("="*100 + "\n")
            f.write(f"Total de erros: {stats['lemma_errors']['total']}\n\n")

            if stats['lemma_errors']['by_pos']:
                f.write("Erros por classe POS:\n")
                for pos_tag, count in stats['lemma_errors']['by_pos']:
                    f.write(f"  {pos_tag}: {count} erros\n")

            f.write("\nExemplos de erros (primeiros 20):\n")
            f.write("-" * 100 + "\n")
            for i, error in enumerate(self.lemma_errors[:20], 1):
                f.write(f"\nErro {i}:\n")
                f.write(f"  Sentença: {error['text']}\n")
                f.write(f"  Token: '{error['token']}' (posição {error['position']})\n")
                f.write(f"  POS: {error['pos_tag']}\n")
                f.write(f"  Gold Lemma: {error['gold_lemma']}\n")
                f.write(f"  Predicted Lemma: {error['predicted_lemma']}\n")
                f.write(f"  Modelo: {error['model']}\n")

            # Dependency Errors
            f.write("\n\n" + "="*100 + "\n")
            f.write("3. ERROS DE PARSING DE DEPENDÊNCIAS\n")
            f.write("="*100 + "\n")
            f.write(f"Total de erros: {stats['dep_errors']['total']}\n")
            f.write(f"  - Apenas HEAD incorreto: {stats['dep_errors']['head_only_errors']}\n")
            f.write(f"  - Apenas DEPREL incorreto: {stats['dep_errors']['deprel_only_errors']}\n")
            f.write(f"  - Ambos incorretos: {stats['dep_errors']['both_errors']}\n\n")

            f.write("Exemplos de erros (primeiros 20):\n")
            f.write("-" * 100 + "\n")
            for i, error in enumerate(self.dep_errors[:20], 1):
                f.write(f"\nErro {i}:\n")
                f.write(f"  Sentença: {error['text']}\n")
                f.write(f"  Token: '{error['token']}' (posição {error['position']}, POS: {error['pos_tag']})\n")
                f.write(f"  Gold: HEAD={error['gold_head']}, DEPREL={error['gold_deprel']}\n")
                f.write(f"  Predicted: HEAD={error['predicted_head']}, DEPREL={error['predicted_deprel']}\n")
                f.write(f"  Modelo: {error['model']}\n")

            # NER Errors
            if self.ner_errors:
                f.write("\n\n" + "="*100 + "\n")
                f.write("4. ERROS DE NER (RECONHECIMENTO DE ENTIDADES)\n")
                f.write("="*100 + "\n")
                f.write(f"Total de erros: {stats['ner_errors']['total']}\n\n")

                if stats['ner_errors']['most_common']:
                    f.write("Erros mais comuns (Gold → Predicted):\n")
                    for error_type, count in stats['ner_errors']['most_common']:
                        f.write(f"  {error_type}: {count} ocorrências\n")

                f.write("\nExemplos de erros (primeiros 20):\n")
                f.write("-" * 100 + "\n")
                for i, error in enumerate(self.ner_errors[:20], 1):
                    f.write(f"\nErro {i}:\n")
                    f.write(f"  Sentença: {error['text']}\n")
                    f.write(f"  Token: '{error['token']}' (posição {error['position']})\n")
                    f.write(f"  Gold Tag: {error['gold_tag']}\n")
                    f.write(f"  Predicted Tag: {error['predicted_tag']}\n")
                    f.write(f"  Modelo: {error['model']}\n")

            f.write("\n\n" + "="*100 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*100 + "\n")

    def save_statistics_json(self, output_path: str):
        """Salva estatísticas em JSON."""
        stats = self.get_error_statistics()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)