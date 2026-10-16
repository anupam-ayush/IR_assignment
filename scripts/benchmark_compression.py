#!/usr/bin/env python3
"""Benchmark index compression algorithms on top-100 frequent vs rarest 10,000 terms."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.index.compression import (
    GapEncoder,
    VariableByteEncoder,
    GammaEncoder,
    UncompressedEncoder,
)
from lexis.evaluation.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Benchmark Compression Schemes")
    parser.add_argument("--index", default="outputs/index", help="Path to index directory")
    parser.add_argument("--output", default="outputs/benchmarks/compression.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"[*] Loading index from {args.index}...")
    index = PositionalIndex(args.index)
    index.load(args.index)

    vocab = index.vocabulary()
    if not vocab:
        print("[!] Index has empty vocabulary!")
        return

    # Sort terms by document frequency
    term_dfs = [(t, index.dictionary.get(t).doc_frequency) for t in vocab]
    term_dfs.sort(key=lambda x: -x[1])

    top_100 = [t for t, _ in term_dfs[:100]]
    rarest_10000 = [t for t, _ in term_dfs[-10000:]] if len(term_dfs) >= 10000 else [t for t, _ in term_dfs[100:]]

    groups = {
        "100_most_frequent": top_100,
        "rarest_10000": rarest_10000,
    }

    rows = []

    print("\n" + "=" * 80)
    print("COMPRESSION BENCHMARK RESULTS")
    print("=" * 80)

    for group_name, terms in groups.items():
        print(f"\n[*] Evaluating group: {group_name} ({len(terms)} terms)...")

        # Accumulate doc_id postings lists
        all_doc_lists = []
        for t in terms:
            plist = index.get_postings(t)
            if plist and plist.postings:
                all_doc_lists.append(plist.doc_ids())

        if not all_doc_lists:
            continue

        # 1. Uncompressed (32-bit uint)
        uncompressed_bytes = sum(len(UncompressedEncoder.encode(ids)) for ids in all_doc_lists)

        # 2. Gap encoding alone (gaps stored as 32-bit ints)
        gap_bytes = sum(len(UncompressedEncoder.encode(GapEncoder.encode(ids))) for ids in all_doc_lists)

        # 3. Variable-Byte encoding on gaps
        vb_bytes = sum(len(VariableByteEncoder.encode(GapEncoder.encode(ids))) for ids in all_doc_lists)

        # 4. Gamma encoding on gaps
        gamma_bytes = sum(len(GammaEncoder.encode(GapEncoder.encode(ids))) for ids in all_doc_lists)

        schemes = [
            ("uncompressed", uncompressed_bytes),
            ("gap_only", gap_bytes),
            ("variable_byte", vb_bytes),
            ("gamma", gamma_bytes),
        ]

        for s_name, s_bytes in schemes:
            ratio = uncompressed_bytes / float(s_bytes) if s_bytes > 0 else 1.0
            space_saved = ((uncompressed_bytes - s_bytes) / float(uncompressed_bytes) * 100.0) if uncompressed_bytes > 0 else 0.0

            print(f"    {s_name:<15}: {s_bytes:>10,} bytes | Ratio: {ratio:5.2f}x | Saved: {space_saved:5.1f}%")

            rows.append({
                "term_group": group_name,
                "scheme": s_name,
                "bytes": s_bytes,
                "compression_ratio": round(ratio, 3),
                "space_saving_pct": round(space_saved, 2),
            })

    fieldnames = ["term_group", "scheme", "bytes", "compression_ratio", "space_saving_pct"]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"\n[OK] Compression benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
