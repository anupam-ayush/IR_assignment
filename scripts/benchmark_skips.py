#!/usr/bin/env python3
"""Benchmark skip-pointer intersection efficiency on 8 term pairs across 3 skip spacings."""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.index.skip_pointers import (
    intersect_without_skips,
    intersect_with_skips,
)
from lexis.evaluation.reports import ReportGenerator

# 8 representative term pairs with varying posting lengths
TERM_PAIRS = [
    ("oil", "opec"),
    ("wheat", "corn"),
    ("bank", "dollar"),
    ("trade", "deficit"),
    ("interest", "rates"),
    ("japan", "korea"),
    ("gold", "silver"),
    ("coffee", "brazil"),
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Skip Pointers Efficiency")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--output", default="outputs/benchmarks/skips.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)

    spacings = [4, 8, 16]
    rows = []

    print("\n" + "=" * 95)
    print(f"{'Term Pair':<20} {'Spacing':<8} {'No Skip Comp':<14} {'With Skip Comp':<15} {'Saved':<8} {'Skips Taken':<12}")
    print("=" * 95)

    for t1, t2 in TERM_PAIRS:
        p1 = index.get_postings(t1)
        p2 = index.get_postings(t2)

        if not p1 or not p2:
            continue

        ids1 = p1.doc_ids()
        ids2 = p2.doc_ids()

        # Intersect without skips baseline
        t0 = time.perf_counter()
        _, stats_no_skip = intersect_without_skips(ids1, ids2)
        base_time = (time.perf_counter() - t0) * 1000.0

        for s in spacings:
            t0 = time.perf_counter()
            _, stats_skip = intersect_with_skips(ids1, ids2, spacing1=s, spacing2=s)
            skip_time = (time.perf_counter() - t0) * 1000.0

            saved = stats_no_skip.comparisons - stats_skip.comparisons
            pct_saved = (saved / max(1, stats_no_skip.comparisons)) * 100.0

            pair_str = f"{t1} & {t2}"
            print(f"{pair_str:<20} {s:<8} {stats_no_skip.comparisons:<14} {stats_skip.comparisons:<15} {saved:<8} {stats_skip.skips_taken:<12}")

            rows.append({
                "term_1": t1,
                "term_2": t2,
                "len_1": len(ids1),
                "len_2": len(ids2),
                "spacing": s,
                "comparisons_no_skip": stats_no_skip.comparisons,
                "comparisons_with_skip": stats_skip.comparisons,
                "comparisons_saved": saved,
                "comparisons_saved_pct": round(pct_saved, 2),
                "skips_taken": stats_skip.skips_taken,
                "time_no_skip_ms": round(base_time, 3),
                "time_skip_ms": round(skip_time, 3),
            })

    print("=" * 95)
    fieldnames = [
        "term_1",
        "term_2",
        "len_1",
        "len_2",
        "spacing",
        "comparisons_no_skip",
        "comparisons_with_skip",
        "comparisons_saved",
        "comparisons_saved_pct",
        "skips_taken",
        "time_no_skip_ms",
        "time_skip_ms",
    ]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"\n[OK] Skip pointers benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
