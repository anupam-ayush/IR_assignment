#!/usr/bin/env python3
"""Build Lexis inverted index over the corpus."""

import argparse
import sys
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
    parser = argparse.ArgumentParser(description="Build Lexis Inverted Index")
    parser.add_argument("--corpus", default="corpus", help="Path to corpus directory")
    parser.add_argument("--output", default="outputs/index", help="Output directory for index")
    parser.add_argument("--method", choices=["spimi", "bsbi"], default="spimi", help="Indexing algorithm")
    parser.add_argument("--memory-limit", type=int, default=50000, help="Memory limit in tokens before flushing")
    parser.add_argument("--shards", type=int, default=1, help="Number of shards for parallel build (e.g. 1 or 4)")
    parser.add_argument("--stemming", action="store_true", default=True, help="Enable Porter stemming")
    parser.add_argument("--no-stemming", dest="stemming", action="store_false", help="Disable Porter stemming")
    parser.add_argument("--stopwords", action="store_true", default=True, help="Enable stopword removal")
    parser.add_argument("--no-stopwords", dest="stopwords", action="store_false", help="Disable stopword removal")
    parser.add_argument("--report", default="outputs/benchmarks/build_metrics.json", help="Path to output metrics JSON")
    args = parser.parse_args()

    tokenizer = Tokenizer(
        normalizer=Normalizer(),
        stemmer=PorterStemmer(enabled=args.stemming),
        stopword_filter=StopwordFilter(enabled=args.stopwords),
    )

    print(f"[*] Starting index build...")
    print(f"    Corpus:       {args.corpus}")
    print(f"    Output:       {args.output}")
    print(f"    Method:       {args.method.upper()}")
    print(f"    Memory Limit: {args.memory_limit} tokens")
    print(f"    Shards:       {args.shards}")
    print(f"    Stemming:     {args.stemming}")
    print(f"    Stopwords:    {args.stopwords}")

    if args.shards > 1:
        builder = ShardBuilder(tokenizer, num_shards=args.shards)
        index, elapsed = builder.build_sharded_index(args.corpus, args.output)
        metrics_data = {
            "method": f"Parallel-Shard ({args.shards} shards)",
            "build_time_seconds": round(elapsed, 3),
            "total_docs": index.doc_count(),
            "vocab_size": index.dictionary.size(),
        }
    elif args.method == "spimi":
        indexer = SPIMIIndexer(tokenizer, memory_limit_tokens=args.memory_limit)
        index, metrics = indexer.build_index(args.corpus, args.output)
        metrics_data = {
            "method": metrics.method,
            "num_blocks": metrics.num_blocks,
            "bytes_written": metrics.bytes_written,
            "peak_memory_mb": metrics.peak_memory_mb,
            "build_time_seconds": metrics.build_time_seconds,
            "total_docs": metrics.total_docs,
            "vocab_size": metrics.vocab_size,
        }
    else:
        indexer = BSBIIndexer(tokenizer, memory_limit_tokens=args.memory_limit)
        index, metrics = indexer.build_index(args.corpus, args.output)
        metrics_data = {
            "method": metrics.method,
            "num_blocks": metrics.num_blocks,
            "bytes_written": metrics.bytes_written,
            "peak_memory_mb": metrics.peak_memory_mb,
            "build_time_seconds": metrics.build_time_seconds,
            "total_docs": metrics.total_docs,
            "vocab_size": metrics.vocab_size,
        }

    ReportGenerator.write_json(args.report, metrics_data)
    print(f"[OK] Index successfully built in {args.output}")
    print(f"    Total Docs: {metrics_data['total_docs']}, Vocab: {metrics_data['vocab_size']}")
    print(f"    Time: {metrics_data['build_time_seconds']}s")
    print(f"    Report saved to {args.report}")


if __name__ == "__main__":
    main()
