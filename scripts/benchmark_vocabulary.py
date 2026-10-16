#!/usr/bin/env python3
"""Benchmark vocabulary properties: Heaps' Law, Zipf's Law, and Plain vs Positional Index Size."""

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.indexing.document_reader import DocumentReader
from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.index.positional_index import PositionalIndex
from lexis.index.compression import GapEncoder, VariableByteEncoder, UncompressedEncoder
from lexis.evaluation.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vocabulary Distributions and Plain vs Positional Index")
    parser.add_argument("--corpus", default="corpus", help="Path to corpus directory")
    parser.add_argument("--index", default="outputs/index", help="Path to existing built positional index")
    parser.add_argument("--vocab-output", default="outputs/benchmarks/vocabulary_fits.csv", help="Vocabulary CSV output")
    parser.add_argument("--plain-output", default="outputs/benchmarks/plain_vs_positional.csv", help="Plain vs Positional CSV output")
    args = parser.parse_args()

    tokenizer = Tokenizer(
        normalizer=Normalizer(),
        stemmer=PorterStemmer(enabled=True),
        stopword_filter=StopwordFilter(enabled=True),
    )

    print("=" * 80)
    print("1. VOCABULARY GROWTH & HEAPS' LAW FITTING")
    print("=" * 80)

    token_counts: List[int] = []
    vocab_sizes: List[int] = []
    seen_vocab = set()
    total_tokens = 0
    term_frequencies: Counter = Counter()

    print("[*] Streaming documents to record vocabulary growth...")
    doc_num = 0
    checkpoint_step = 250

    for doc in DocumentReader.stream_corpus(args.corpus):
        tokens = tokenizer.tokenize_document(title=doc.title, body=doc.body)
        for t in tokens:
            term = t.term
            seen_vocab.add(term)
            term_frequencies[term] += 1

        total_tokens += len(tokens)
        doc_num += 1

        if doc_num % checkpoint_step == 0:
            token_counts.append(total_tokens)
            vocab_sizes.append(len(seen_vocab))

    # Add final point if not already added
    if not token_counts or token_counts[-1] != total_tokens:
        token_counts.append(total_tokens)
        vocab_sizes.append(len(seen_vocab))

    k, b = ReportGenerator.fit_heaps_law(token_counts, vocab_sizes)
    print(f"[*] Total Documents: {doc_num:,}")
    print(f"[*] Total Tokens (T): {total_tokens:,}")
    print(f"[*] Unique Vocabulary (V): {len(seen_vocab):,}")
    print(f"\n[+] Empirical Heaps' Law Fit (V = k * T^b):")
    print(f"    Empirical (Reuters-21578): k = {k}, b = {b}")
    print(f"    Textbook (Reuters-RCV1):   k ~ 44, b ~ 0.49  (Manning et al., Ch 5, Table 5.1)")

    print("\n" + "=" * 80)
    print("2. ZIPF'S LAW FITTING")
    print("=" * 80)

    most_common = term_frequencies.most_common()
    ranks = list(range(1, len(most_common) + 1))
    frequencies = [freq for _, freq in most_common]

    c, alpha = ReportGenerator.fit_zipfs_law(ranks, frequencies)
    print(f"[+] Empirical Zipf's Law Fit (f = c / r^alpha):")
    print(f"    Empirical (Reuters-21578): c = {c}, alpha = {alpha}")
    print(f"    Textbook (Zipf Ideal):     c ~ constant, alpha ~ 1.00  (Manning et al., Ch 5)")

    print(f"\nTop 10 Most Frequent Stems:")
    for rank, (term, count) in enumerate(most_common[:10], start=1):
        print(f"    {rank:>2}. {term:<15} freq: {count:>6,}")

    vocab_rows = [
        {
            "law": "Heaps' Law (V = k * T^b)",
            "metric": "k",
            "empirical_value": k,
            "textbook_rcv1_value": 44.0,
            "notes": "Reuters-21578 vs Reuters-RCV1 (Manning Table 5.1)",
        },
        {
            "law": "Heaps' Law (V = k * T^b)",
            "metric": "b",
            "empirical_value": b,
            "textbook_rcv1_value": 0.49,
            "notes": "Sublinear vocabulary growth exponent",
        },
        {
            "law": "Zipf's Law (f = c / r^alpha)",
            "metric": "alpha",
            "empirical_value": alpha,
            "textbook_rcv1_value": 1.00,
            "notes": "Power-law rank-frequency decay exponent",
        },
        {
            "law": "Zipf's Law (f = c / r^alpha)",
            "metric": "c",
            "empirical_value": c,
            "textbook_rcv1_value": "N/A",
            "notes": "Proportionality constant",
        },
    ]
    ReportGenerator.write_csv(
        args.vocab_output,
        ["law", "metric", "empirical_value", "textbook_rcv1_value", "notes"],
        vocab_rows,
    )
    print(f"\n[OK] Vocabulary benchmark results saved to {args.vocab_output}")

    print("\n" + "=" * 80)
    print("3. POSITIONAL INDEX SIZE AGAINST PLAIN INDEX")
    print("=" * 80)

    # Load built positional index
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"[!] Index path {args.index} does not exist. Skipping plain index size comparison.")
        return

    index = PositionalIndex(str(index_path))
    index.load(str(index_path))

    # Calculate actual positional postings size (postings.dat on disk)
    postings_dat_path = index_path / "postings.dat"
    positional_disk_bytes = postings_dat_path.stat().st_size if postings_dat_path.exists() else 0

    # Calculate uncompressed and gap-encoded plain (non-positional) index sizes
    # A plain index only stores doc_ids (and tf), without word positions
    total_postings_entries = 0
    total_positions_stored = 0

    plain_uncompressed_bytes = 0
    plain_gap_vb_bytes = 0

    positional_raw_bytes = 0

    for term in index.vocabulary():
        plist = index.get_postings(term)
        if not plist:
            continue

        doc_ids = plist.doc_ids()
        total_postings_entries += len(doc_ids)

        # Plain uncompressed (32-bit doc_id per entry)
        plain_uncompressed_bytes += len(doc_ids) * 4

        # Plain compressed (Variable Byte on doc_id gaps)
        gaps = GapEncoder.encode(doc_ids)
        plain_gap_vb_bytes += len(VariableByteEncoder.encode(gaps))

        # Positional entries: doc_id (4 bytes) + tf (2 bytes) + position array (4 bytes each)
        for p in plist.postings:
            total_positions_stored += len(p.positions)
            positional_raw_bytes += 4 + 2 + (len(p.positions) * 4)

    ratio_uncompressed = positional_raw_bytes / max(1, plain_uncompressed_bytes)
    ratio_disk_vs_plain_vb = positional_disk_bytes / max(1, plain_gap_vb_bytes)

    print(f"[*] Total Postings (term, doc) pairs: {total_postings_entries:,}")
    print(f"[*] Total Word Positions stored:      {total_positions_stored:,}")
    print(f"[*] Plain Index (Uncompressed doc_ids):       {plain_uncompressed_bytes:>10,} bytes ({plain_uncompressed_bytes / (1024*1024):.2f} MB)")
    print(f"[*] Plain Index (Gap + Variable-Byte):        {plain_gap_vb_bytes:>10,} bytes ({plain_gap_vb_bytes / (1024*1024):.2f} MB)")
    print(f"[*] Positional Index (Raw binary with pos):   {positional_raw_bytes:>10,} bytes ({positional_raw_bytes / (1024*1024):.2f} MB)")
    print(f"[*] Positional Index (On-Disk zlib/postings): {positional_disk_bytes:>10,} bytes ({positional_disk_bytes / (1024*1024):.2f} MB)")
    print(f"\n[+] Positional vs Plain Size Ratio (Uncompressed): {ratio_uncompressed:.2f}x")
    print(f"[+] Positional On-Disk vs Plain Gap-VB Ratio:      {ratio_disk_vs_plain_vb:.2f}x")

    plain_rows = [
        {
            "index_type": "Plain Non-Positional (Uncompressed)",
            "description": "Stores only doc_id per posting (32-bit uint)",
            "bytes": plain_uncompressed_bytes,
            "size_mb": round(plain_uncompressed_bytes / (1024 * 1024), 2),
            "ratio_to_plain_uncompressed": 1.0,
        },
        {
            "index_type": "Plain Non-Positional (Gap + Variable-Byte)",
            "description": "Stores doc_id gaps with Variable-Byte encoding",
            "bytes": plain_gap_vb_bytes,
            "size_mb": round(plain_gap_vb_bytes / (1024 * 1024), 2),
            "ratio_to_plain_uncompressed": round(plain_gap_vb_bytes / plain_uncompressed_bytes, 3),
        },
        {
            "index_type": "Positional Index (Raw with word offsets)",
            "description": "Stores doc_id, tf, zone, and full word positions array",
            "bytes": positional_raw_bytes,
            "size_mb": round(positional_raw_bytes / (1024 * 1024), 2),
            "ratio_to_plain_uncompressed": round(ratio_uncompressed, 3),
        },
        {
            "index_type": "Positional Index (On-Disk postings.dat)",
            "description": "Serialized and compressed positional postings file",
            "bytes": positional_disk_bytes,
            "size_mb": round(positional_disk_bytes / (1024 * 1024), 2),
            "ratio_to_plain_uncompressed": round(positional_disk_bytes / plain_uncompressed_bytes, 3),
        },
    ]

    ReportGenerator.write_csv(
        args.plain_output,
        ["index_type", "description", "bytes", "size_mb", "ratio_to_plain_uncompressed"],
        plain_rows,
    )
    print(f"\n[OK] Plain vs Positional benchmark saved to {args.plain_output}")


if __name__ == "__main__":
    main()
