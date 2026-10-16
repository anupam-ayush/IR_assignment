#!/usr/bin/env python3
"""Run ablation experiments: disable stemming, stop words, IDF, length norm, and zone weighting one-at-a-time."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.ranking.scorer import RankedScorer
from lexis.evaluation.ranking_metrics import RankingMetrics
from lexis.evaluation.significance import SignificanceTester
from lexis.evaluation.reports import ReportGenerator


def load_queries(tsv_path: str) -> Dict[str, str]:
    queries = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] != "query_id":
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(tsv_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    rel_map: Dict[str, Set[str]] = {}
    grades_map: Dict[str, Dict[str, int]] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] != "query_id":
                qid, doc, grade = parts[0], parts[1], int(parts[2])
                if qid not in grades_map:
                    grades_map[qid] = {}
                    rel_map[qid] = set()
                grades_map[qid][doc] = grade
                if grade > 0:
                    rel_map[qid].add(doc)
    return rel_map, grades_map


def run_system_ap(
    index: PositionalIndex,
    tokenizer: Tokenizer,
    queries: Dict[str, str],
    rel_docs_map: Dict[str, Set[str]],
    use_idf: bool = True,
    use_zones: bool = True,
    use_length_norm: bool = True,
) -> Tuple[float, Dict[str, float]]:
    scorer = RankedScorer(
        index=index,
        tokenizer=tokenizer,
        use_idf=use_idf,
        use_zones=use_zones,
        use_length_norm=use_length_norm,
    )

    query_retrievals = {}
    for qid, q_text in queries.items():
        results, _, _ = scorer.score_query(q_text, top_k=100)
        query_retrievals[qid] = [r.doc_name for r in results]

    mean_ap, per_query_ap = RankingMetrics.mean_average_precision(query_retrievals, rel_docs_map)
    return mean_ap, per_query_ap


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Experiments with Statistical Significance")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--queries", default="queries_train.tsv", help="Path to queries TSV")
    parser.add_argument("--qrels", default="qrels.tsv", help="Path to qrels.tsv")
    parser.add_argument("--output", default="outputs/evaluation/ablations.csv", help="Ablations CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)

    queries = load_queries(args.queries)
    rel_docs_map, _ = load_qrels(args.qrels)

    print(f"[*] Loaded {len(queries)} queries for ablation study.")

    # 1. Full System
    full_tok = Tokenizer(
        normalizer=Normalizer(),
        stemmer=PorterStemmer(enabled=True),
        stopword_filter=StopwordFilter(enabled=True),
    )
    print("\n[*] Evaluating Full System...")
    full_map, full_ap_dict = run_system_ap(
        index=index,
        tokenizer=full_tok,
        queries=queries,
        rel_docs_map=rel_docs_map,
        use_idf=True,
        use_zones=True,
        use_length_norm=True,
    )
    print(f"    Full System MAP: {full_map:.4f}")

    # Ablations configurations to test one-by-one
    ablations = [
        ("no_stemming", PorterStemmer(enabled=False), StopwordFilter(enabled=True), True, True, True),
        ("no_stopwords", PorterStemmer(enabled=True), StopwordFilter(enabled=False), True, True, True),
        ("no_idf", PorterStemmer(enabled=True), StopwordFilter(enabled=True), False, True, True),
        ("no_length_norm", PorterStemmer(enabled=True), StopwordFilter(enabled=True), True, True, False),
        ("no_zone_weighting", PorterStemmer(enabled=True), StopwordFilter(enabled=True), True, False, True),
    ]

    rows = [{
        "configuration": "full_system",
        "map": round(full_map, 4),
        "delta_map": 0.0,
        "p_value": 1.0,
        "statistically_significant_05": False,
    }]

    print("\n" + "=" * 80)
    print(f"{'Configuration':<22} {'MAP':<10} {'Delta MAP':<12} {'p-value':<10} {'Sig (p<0.05)'}")
    print("=" * 80)
    print(f"{'full_system':<22} {full_map:<10.4f} {'0.0000':<12} {'1.0000':<10} {'NO'}")

    for name, stemmer, sw_filter, use_idf, use_zones, use_norm in ablations:
        tok = Tokenizer(normalizer=Normalizer(), stemmer=stemmer, stopword_filter=sw_filter)
        ablated_map, ablated_ap_dict = run_system_ap(
            index=index,
            tokenizer=tok,
            queries=queries,
            rel_docs_map=rel_docs_map,
            use_idf=use_idf,
            use_zones=use_zones,
            use_length_norm=use_norm,
        )

        sig = SignificanceTester.paired_t_test(full_ap_dict, ablated_ap_dict)
        sig_str = "YES" if sig.is_significant_05 else "NO"

        print(f"{name:<22} {ablated_map:<10.4f} {sig.delta_map:<12.4f} {sig.p_value:<10.4f} {sig_str}")

        rows.append({
            "configuration": name,
            "map": round(ablated_map, 4),
            "delta_map": sig.delta_map,
            "p_value": sig.p_value,
            "statistically_significant_05": sig.is_significant_05,
        })

    print("=" * 80)
    fieldnames = ["configuration", "map", "delta_map", "p_value", "statistically_significant_05"]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"\n[OK] Ablation results saved to {args.output}")


if __name__ == "__main__":
    main()
