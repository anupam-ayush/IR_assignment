#!/usr/bin/env python3
"""Evaluate ranking models on training queries: P@10, MAP, NDCG@10, and 11-point PR curve."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.ranking.scorer import RankedScorer
from lexis.evaluation.ranking_metrics import RankingMetrics
from lexis.evaluation.precision_recall import PrecisionRecallCurve
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
    """Returns (relevant_docs_per_query, grades_per_query_doc)."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ranking on Training Queries")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--queries", default="queries_train.tsv", help="Path to training queries TSV")
    parser.add_argument("--qrels", default="qrels.tsv", help="Path to qrels.tsv")
    parser.add_argument("--output", default="outputs/evaluation/train_metrics.csv", help="Metrics CSV path")
    parser.add_argument("--pr-output", default="outputs/evaluation/train_pr_curve.csv", help="11-point PR curve CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)
    tokenizer = Tokenizer()

    queries = load_queries(args.queries)
    rel_docs_map, grades_map = load_qrels(args.qrels)

    print(f"[*] Loaded {len(queries)} queries and qrels for {len(rel_docs_map)} queries.")

    # Configurations:
    # 1. boolean_system: unranked / coordination-level Boolean retrieval
    # 2. raw-TF baseline: use_idf=False, use_zones=False, use_length_norm=False, use_log_tf=False
    # 3. tf-idf baseline: use_idf=True, use_zones=False, use_length_norm=True, use_log_tf=True
    # 4. full system: use_idf=True, use_zones=True, use_length_norm=True, use_log_tf=True
    configs = [
        ("boolean_system", None),
        ("raw_tf_baseline", (False, False, False, False)),
        ("tfidf_baseline", (True, False, True, True)),
        ("full_system", (True, True, True, True)),
    ]

    metric_rows = []
    pr_rows = []

    print("\n" + "=" * 80)
    print("TRAINING EVALUATION RESULTS")
    print("=" * 80)

    from collections import Counter

    for cfg_name, params in configs:
        query_retrievals: Dict[str, List[str]] = {}
        p10_list = []
        ndcg10_list = []

        if cfg_name == "boolean_system":
            for qid, q_text in queries.items():
                terms = tokenizer.tokenize_terms(q_text)
                matching_docs = Counter()
                for t in set(terms):
                    plist = index.get_postings(t)
                    if plist:
                        for did in plist.doc_ids():
                            if not index.document_registry.is_deleted(did):
                                matching_docs[did] += 1
                ranked_docs = sorted(matching_docs.items(), key=lambda x: (-x[1], index.get_doc_name(x[0]) or ""))
                ret_doc_names = [index.get_doc_name(did) for did, _ in ranked_docs[:100] if index.get_doc_name(did)]
                query_retrievals[qid] = ret_doc_names

                rel_docs = rel_docs_map.get(qid, set())
                p10 = RankingMetrics.precision_at_k(ret_doc_names, rel_docs, k=10)
                ndcg10 = RankingMetrics.ndcg_at_k(ret_doc_names, grades_map.get(qid, {}), k=10)
                p10_list.append(p10)
                ndcg10_list.append(ndcg10)
        else:
            use_idf, use_zones, use_norm, use_log_tf = params
            scorer = RankedScorer(
                index=index,
                tokenizer=tokenizer,
                use_idf=use_idf,
                use_zones=use_zones,
                use_length_norm=use_norm,
                use_log_tf=use_log_tf,
            )

            for qid, q_text in queries.items():
                results, _, _ = scorer.score_query(q_text, top_k=100)
                ret_doc_names = [r.doc_name for r in results]
                query_retrievals[qid] = ret_doc_names

                rel_docs = rel_docs_map.get(qid, set())
                p10 = RankingMetrics.precision_at_k(ret_doc_names, rel_docs, k=10)
                ndcg10 = RankingMetrics.ndcg_at_k(ret_doc_names, grades_map.get(qid, {}), k=10)

                p10_list.append(p10)
                ndcg10_list.append(ndcg10)

        mean_p10 = sum(p10_list) / max(1, len(p10_list))
        mean_ndcg10 = sum(ndcg10_list) / max(1, len(ndcg10_list))
        mean_ap, _ = RankingMetrics.mean_average_precision(query_retrievals, rel_docs_map)

        print(f"\nConfiguration: {cfg_name}")
        print(f"    Precision@10: {mean_p10:.4f}")
        print(f"    MAP:          {mean_ap:.4f}")
        print(f"    NDCG@10:      {mean_ndcg10:.4f}")

        metric_rows.append({
            "model": cfg_name,
            "precision_at_10": round(mean_p10, 4),
            "map": round(mean_ap, 4),
            "ndcg_at_10": round(mean_ndcg10, 4),
        })

        # Compute 11-point PR curve for full system
        if cfg_name == "full_system":
            pr_curve = PrecisionRecallCurve.compute_aggregate(query_retrievals, rel_docs_map)
            for r_lvl, prec in pr_curve:
                pr_rows.append({"recall_level": r_lvl, "interpolated_precision": prec})

    ReportGenerator.write_csv(args.output, ["model", "precision_at_10", "map", "ndcg_at_10"], metric_rows)
    if pr_rows:
        ReportGenerator.write_csv(args.pr_output, ["recall_level", "interpolated_precision"], pr_rows)

    print(f"\n[OK] Training metrics saved to {args.output}")
    print(f"[OK] 11-point PR curve saved to {args.pr_output}")


if __name__ == "__main__":
    main()
