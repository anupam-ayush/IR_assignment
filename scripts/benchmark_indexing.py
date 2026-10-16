#!/usr/bin/env python3
"""Benchmark indexing construction: memory limits, SPIMI vs BSBI, and parallel shards."""

import argparse
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.indexing.spimi import SPIMIIndexer
from lexis.indexing.bsbi import BSBIIndexer
from lexis.indexing.shard_builder import ShardBuilder
from lexis.evaluation.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Benchmark Index Construction Algorithms")
    parser.add_argument("--corpus", default="corpus", help="Path to corpus directory")
    parser.add_argument("--output", default="outputs/benchmarks/indexing.csv", help="Output CSV path")
    parser.add_argument("--temp-dir", default="outputs/temp_benchmark", help="Temporary directory for partial blocks")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional max docs to index (for quick validation)")
    args = parser.parse_args()

    tokenizer = Tokenizer(
        normalizer=Normalizer(),
        stemmer=PorterStemmer(enabled=True),
        stopword_filter=StopwordFilter(enabled=True),
    )

    temp_path = Path(args.temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    rows = []
    memory_limits = [10000, 25000, 50000, 100000]

    print("=" * 80)
    print("1. SPIMI MEMORY LIMIT EXPERIMENTS")
    print("=" * 80)
    for mem in memory_limits:
        print(f"[*] Running SPIMI with memory limit = {mem} tokens...")
        out_dir = temp_path / f"spimi_{mem}"
        indexer = SPIMIIndexer(tokenizer, memory_limit_tokens=mem, temp_dir=str(temp_path))
        _, metrics = indexer.build_index(args.corpus, str(out_dir), max_docs=args.max_docs)

        print(f"    Blocks: {metrics.num_blocks}, Bytes: {metrics.bytes_written:,}, Peak Mem: {metrics.peak_memory_mb} MB, Time: {metrics.build_time_seconds}s")
        rows.append({
            "experiment": "memory_limit_spimi",
            "method": "SPIMI",
            "memory_limit_tokens": mem,
            "shards": 1,
            "num_blocks": metrics.num_blocks,
            "bytes_written": metrics.bytes_written,
            "peak_memory_mb": metrics.peak_memory_mb,
            "build_time_seconds": metrics.build_time_seconds,
            "speedup": 1.0,
        })

    print("\n" + "=" * 80)
    print("2. BSBI COMPARISON EXPERIMENT")
    print("=" * 80)
    for mem in [25000, 50000]:
        print(f"[*] Running BSBI with memory limit = {mem} tokens...")
        out_dir = temp_path / f"bsbi_{mem}"
        indexer = BSBIIndexer(tokenizer, memory_limit_tokens=mem, temp_dir=str(temp_path))
        _, metrics = indexer.build_index(args.corpus, str(out_dir), max_docs=args.max_docs)

        print(f"    Blocks: {metrics.num_blocks}, Bytes: {metrics.bytes_written:,}, Peak Mem: {metrics.peak_memory_mb} MB, Time: {metrics.build_time_seconds}s")
        rows.append({
            "experiment": "spimi_vs_bsbi",
            "method": "BSBI",
            "memory_limit_tokens": mem,
            "shards": 1,
            "num_blocks": metrics.num_blocks,
            "bytes_written": metrics.bytes_written,
            "peak_memory_mb": metrics.peak_memory_mb,
            "build_time_seconds": metrics.build_time_seconds,
            "speedup": 1.0,
        })

    print("\n" + "=" * 80)
    print("3. PARALLEL SHARDING EXPERIMENT (1, 2, 4 SHARDS)")
    print("=" * 80)
    base_time = None
    for shards in [1, 2, 4]:
        print(f"[*] Running Parallel Indexing with {shards} shard(s)...")
        out_dir = temp_path / f"shards_{shards}"
        builder = ShardBuilder(tokenizer, num_shards=shards)
        _, elapsed = builder.build_sharded_index(args.corpus, str(out_dir))

        if shards == 1:
            base_time = elapsed
            speedup = 1.0
        else:
            speedup = round((base_time / elapsed) if elapsed > 0 else 1.0, 2)

        print(f"    Wall-Clock Time: {elapsed:.2f}s (Speedup: {speedup}x)")
        rows.append({
            "experiment": "parallel_shards",
            "method": f"Parallel ({shards}-shard)",
            "memory_limit_tokens": 0,
            "shards": shards,
            "num_blocks": shards,
            "bytes_written": 0,
            "peak_memory_mb": 0.0,
            "build_time_seconds": round(elapsed, 3),
            "speedup": speedup,
        })

    # Clean up temp directory
    try:
        shutil.rmtree(temp_path)
    except Exception:
        pass

    fieldnames = [
        "experiment",
        "method",
        "memory_limit_tokens",
        "shards",
        "num_blocks",
        "bytes_written",
        "peak_memory_mb",
        "build_time_seconds",
        "speedup",
    ]
    ReportGenerator.write_csv(args.output, fieldnames, rows)
    print(f"\n[OK] Indexing benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
