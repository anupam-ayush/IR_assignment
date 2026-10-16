#!/usr/bin/env python3
"""Run validation on boolean_tests.tsv covering Boolean, phrase, wildcard, and proximity."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine
from lexis.evaluation.boolean_eval import BooleanEvaluatorRunner
from lexis.evaluation.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Validate Boolean and Exact Retrieval Tests")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--tests", default="boolean_tests.tsv", help="Path to boolean_tests.tsv")
    parser.add_argument("--output", default="outputs/evaluation/boolean_results.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)

    # For exact retrieval validation against brute-force scan, stemming & stopword filtering are disabled
    tokenizer = Tokenizer(
        normalizer=Normalizer(case_folding=True),
        stemmer=PorterStemmer(enabled=False),
        stopword_filter=StopwordFilter(enabled=False),
    )

    engine = QueryEngine(index, tokenizer)
    runner = BooleanEvaluatorRunner(engine)

    print(f"[*] Running exact retrieval test suite from {args.tests}...")
    results = runner.run_tests_from_file(args.tests)

    rows = []
    exact_matches = 0

    print(f"\n{'ID':<5} {'Type':<10} {'Expected':<10} {'Actual':<10} {'Match':<8} {'F1':<8} {'Query'}")
    print("-" * 80)

    for r in results:
        if r.exact_match:
            exact_matches += 1

        match_str = "PASS" if r.exact_match else "FAIL"
        print(f"{r.query_id:<5} {r.test_type:<10} {r.expected_count:<10} {r.actual_count:<10} {match_str:<8} {r.f1:<8.4f} {r.query[:35]}")

        rows.append({
            "query_id": r.query_id,
            "query": r.query,
            "type": r.test_type,
            "expected_count": r.expected_count,
            "actual_count": r.actual_count,
            "exact_match": r.exact_match,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "false_positives": r.false_positives,
            "false_negatives": r.false_negatives,
        })

    print("-" * 80)
    print(f"[*] Summary: {exact_matches}/{len(results)} queries matched expected results exactly ({exact_matches/max(1, len(results))*100:.1f}%)")

    fieldnames = list(rows[0].keys()) if rows else []
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"[OK] Full Boolean report saved to {args.output}")


if __name__ == "__main__":
    main()
