#!/usr/bin/env python3
"""Run validation on misspellings.tsv using edit distance and frequency ranking."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.query.spelling import SpellingCorrector
from lexis.evaluation.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Validate Spelling Correction on misspellings.tsv")
    default_idx = "outputs/index_exact" if Path("outputs/index_exact").exists() else "outputs/index"
    parser.add_argument("--index", default=default_idx, help="Path to index directory (prefer unstemmed index_exact)")
    parser.add_argument("--tests", default="misspellings.tsv", help="Path to misspellings.tsv")
    parser.add_argument("--max-distance", type=int, default=2, help="Maximum Levenshtein distance")
    parser.add_argument("--output", default="outputs/evaluation/misspellings_results.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)

    corrector = SpellingCorrector(index)
    print(f"[*] Loaded vocabulary size: {len(corrector.vocab)} terms")

    tests_path = Path(args.tests)
    if not tests_path.exists():
        raise FileNotFoundError(f"File not found: {args.tests}")

    with open(tests_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = []
    correct_count = 0
    total = 0

    print(f"\n{'Misspelled':<18} {'Expected':<18} {'Suggested':<18} {'Status'}")
    print("-" * 65)

    for line in lines[1:]:  # Skip header
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        misspelled = parts[0]
        expected_raw = parts[1]
        accepted_list = [c.strip().lower() for c in expected_raw.split(",")]

        suggested = corrector.suggest_correction(misspelled, max_distance=args.max_distance)
        is_correct = (suggested in accepted_list) if suggested else False
        if is_correct:
            correct_count += 1
        total += 1

        status = "MATCH" if is_correct else "MISMATCH"
        print(f"{misspelled:<18} {expected_raw:<18} {str(suggested):<18} {status}")

        rows.append({
            "misspelled": misspelled,
            "expected": expected_raw,
            "suggested": suggested or "",
            "correct": is_correct,
        })

    print("-" * 65)
    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
    print(f"[*] Summary: {correct_count}/{total} correct suggestions ({accuracy:.1f}% accuracy)")

    fieldnames = ["misspelled", "expected", "suggested", "correct"]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"[OK] Misspelling results saved to {args.output}")


if __name__ == "__main__":
    main()
