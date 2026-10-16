#!/usr/bin/env python3
"""Evaluate Chapter 7 pruning techniques: Champion Lists and Impact Ordering vs Exhaustive."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.ranking.scorer import RankedScorer
from lexis.evaluation.reports import ReportGenerator
from typing import Dict


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pruning Strategies vs Exhaustive Scoring")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--queries", default="queries_train.tsv", help="Path to queries TSV")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results to retrieve")
    parser.add_argument("--output", default="outputs/benchmarks/pruning.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)
    tokenizer = Tokenizer()

    scorer = RankedScorer(index, tokenizer)
    queries = load_queries(args.queries)

    print(f"[*] Pre-building Champion Lists (r=50)...")
    scorer.get_champion_lists(r=50)

    rows = []
    methods = ["exhaustive", "champion_lists", "impact_ordering"]

    # Accumulators for averages
    scored_sums = {m: 0 for m in methods}
    latency_sums = {m: 0.0 for m in methods}
    overlap_sums = {m: 0.0 for m in methods}

    print("\n" + "=" * 90)
    print(f"{'Query ID':<10} {'Method':<18} {'Docs Scored':<14} {'Latency (ms)':<14} {'Overlap@10':<12}")
    print("=" * 90)

    for qid, q_text in queries.items():
        metrics_by_method = scorer.compare_pruning(q_text, top_k=args.top_k)

        for m in methods:
            m_data = metrics_by_method[m]
            scored_sums[m] += m_data.num_scored
            latency_sums[m] += m_data.latency_ms
            overlap_sums[m] += m_data.overlap_at_10

            top_doc_str = ";".join([f"{doc}:{score}" for doc, score in m_data.top_10_results])

            rows.append({
                "query_id": qid,
                "query": q_text,
                "method": m,
                "docs_scored": m_data.num_scored,
                "latency_ms": m_data.latency_ms,
                "overlap_at_10": m_data.overlap_at_10,
                "top_10_results": top_doc_str,
            })

            print(f"{qid:<10} {m:<18} {m_data.num_scored:<14} {m_data.latency_ms:<14.2f} {m_data.overlap_at_10:<12.4f}")

    num_q = max(1, len(queries))
    print("=" * 90)
    print(f"\nAVERAGE PRUNING EFFICIENCY SUMMARY:")
    for m in methods:
        avg_scored = scored_sums[m] / num_q
        avg_lat = latency_sums[m] / num_q
        avg_ovlp = overlap_sums[m] / num_q
        print(f"    {m:<18}: Scored: {avg_scored:6.1f} docs | Latency: {avg_lat:6.2f} ms | Overlap@10: {avg_ovlp:6.4f}")

    fieldnames = [
        "query_id",
        "query",
        "method",
        "docs_scored",
        "latency_ms",
        "overlap_at_10",
        "top_10_results",
    ]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"\n[OK] Pruning benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
