"""Block Sort-Based Indexing (BSBI) implementation."""

import time
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lexis.indexing.document_reader import DocumentReader
from lexis.indexing.block_writer import BlockWriter
from lexis.indexing.spimi import IndexingMetrics
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.index.dictionary import DocumentRegistry
from lexis.preprocessing.tokenizer import Tokenizer


class BSBIIndexer:
    """BSBI Indexer implementation for comparison against SPIMI."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        memory_limit_tokens: int = 50000,
        temp_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.memory_limit_tokens = memory_limit_tokens
        self.temp_dir = Path(temp_dir) if temp_dir else None

    def build_index(
        self,
        corpus_dir: str,
        output_dir: str,
        max_docs: Optional[int] = None,
    ) -> Tuple[PositionalIndex, IndexingMetrics]:
        start_time = time.time()
        process = psutil.Process()
        peak_mem = process.memory_info().rss / (1024 * 1024)

        out_path = Path(output_dir)
        blocks_dir = (self.temp_dir or out_path) / "bsbi_blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        term_to_id: Dict[str, int] = {}
        id_to_term: Dict[int, str] = {}
        registry = DocumentRegistry()

        pairs: List[Tuple[int, int, int, str]] = []  # (term_id, doc_id, pos, zone)
        block_files: List[Path] = []
        bytes_written = 0
        block_idx = 0
        doc_count = 0

        for doc in DocumentReader.stream_corpus(corpus_dir):
            if max_docs is not None and doc_count >= max_docs:
                break

            tokens = self.tokenizer.tokenize_document(title=doc.title, body=doc.body)
            title_len = sum(1 for t in tokens if t.zone == "title")
            body_len = len(tokens) - title_len

            doc_id = registry.register(
                doc_name=doc.doc_name,
                doc_length=len(tokens),
                title_length=title_len,
                body_length=body_len,
            )
            doc_count += 1

            for tok in tokens:
                if tok.term not in term_to_id:
                    tid = len(term_to_id)
                    term_to_id[tok.term] = tid
                    id_to_term[tid] = tok.term
                else:
                    tid = term_to_id[tok.term]

                pairs.append((tid, doc_id, tok.position, tok.zone))

            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_mem:
                peak_mem = current_mem

            if len(pairs) >= self.memory_limit_tokens:
                # In BSBI: sort pairs by (term_id, doc_id)
                pairs.sort(key=lambda x: (x[0], x[1]))
                block_file = blocks_dir / f"bsbi_block_{block_idx:04d}.dat"
                written = BlockWriter.write_bsbi_block(block_file, pairs)
                bytes_written += written
                block_files.append(block_file)
                block_idx += 1
                pairs.clear()

        if pairs:
            pairs.sort(key=lambda x: (x[0], x[1]))
            block_file = blocks_dir / f"bsbi_block_{block_idx:04d}.dat"
            written = BlockWriter.write_bsbi_block(block_file, pairs)
            bytes_written += written
            block_files.append(block_file)
            pairs.clear()

        # Merge BSBI blocks into intermediate SPIMI-like blocks or direct index
        final_index = PositionalIndex(str(out_path))
        final_index.document_registry = registry

        # Group and merge tuples from all blocks efficiently in O(N)
        from collections import defaultdict
        tid_to_entries: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
        for bf in block_files:
            for tid, did, pos, zone in BlockWriter.read_bsbi_block(bf):
                tid_to_entries[tid].append((did, pos, zone))

        all_tids = sorted(term_to_id.values())

        out_path.mkdir(parents=True, exist_ok=True)
        postings_path = out_path / "postings.dat"
        offset = 0
        import json
        import zlib

        with open(postings_path, "wb") as pf:
            for tid in all_tids:
                term = id_to_term[tid]
                entries = tid_to_entries.get(tid)
                if not entries:
                    continue
                plist = PostingsList(term=term)
                entries.sort(key=lambda x: x[0])
                for did, pos, zone in entries:
                    plist.add_occurrence(doc_id=did, position=pos, zone=zone)

                if plist.postings:
                    serialized = json.dumps(plist.to_dict()).encode("utf-8")
                    compressed = zlib.compress(serialized, level=6)
                    length = len(compressed)
                    pf.write(compressed)

                    total_tf = sum(p.tf for p in plist.postings)
                    final_index.dictionary.add_or_update(
                        term=term,
                        doc_frequency=plist.doc_frequency,
                        offset=offset,
                        length=length,
                        total_tf=total_tf,
                    )
                    offset += length

        meta = {
            "dictionary": final_index.dictionary.to_dict(),
            "registry": final_index.document_registry.to_dict(),
        }
        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        build_time = time.time() - start_time
        current_mem = process.memory_info().rss / (1024 * 1024)
        if current_mem > peak_mem:
            peak_mem = current_mem

        metrics = IndexingMetrics(
            method="BSBI",
            num_blocks=len(block_files),
            bytes_written=bytes_written,
            peak_memory_mb=round(peak_mem, 2),
            build_time_seconds=round(build_time, 3),
            total_docs=doc_count,
            vocab_size=final_index.dictionary.size(),
        )

        return final_index, metrics
