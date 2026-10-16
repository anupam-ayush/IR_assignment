"""On-disk and in-memory Positional Inverted Index for Lexis."""

import os
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from lexis.index.dictionary import Dictionary, DocumentRegistry, TermInfo
from lexis.index.postings import Posting, PostingsList


class PositionalIndex:
    """Positional Inverted Index supporting scalable on-disk serialization and on-demand reading."""

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir) if index_dir else None
        self.dictionary = Dictionary()
        self.document_registry = DocumentRegistry()
        self._postings_cache: Dict[str, PostingsList] = {}  # In-memory storage when building
        self._postings_file = None

    def add_term_occurrence(self, term: str, doc_id: int, position: int, zone: str = "body"):
        """Record a term occurrence in a document at a specific position."""
        if term not in self._postings_cache:
            self._postings_cache[term] = PostingsList(term=term)
        self._postings_cache[term].add_occurrence(doc_id, position, zone)
        plist = self._postings_cache[term]
        self.dictionary.add_or_update(
            term=term,
            doc_frequency=plist.doc_frequency,
            total_tf=sum(p.tf for p in plist.postings),
        )

    def get_postings(self, term: str) -> Optional[PostingsList]:
        """Fetch the postings list for a term.
        
        If in memory, returns cached list.
        If on disk, reads directly from postings file using dictionary offset/length.
        Filters out any deleted documents.
        """
        # 1. Check in-memory cache
        plist: Optional[PostingsList] = None
        if term in self._postings_cache:
            plist = self._postings_cache[term]
        elif self.dictionary.contains(term) and self.index_dir:
            # 2. Read from disk
            info = self.dictionary.get(term)
            if info and info.length > 0:
                plist = self._read_postings_from_disk(info.offset, info.length)

        if plist is None:
            return None

        # Filter out deleted documents if any exist
        if self.document_registry.deleted_doc_ids:
            filtered_postings = [
                p for p in plist.postings
                if not self.document_registry.is_deleted(p.doc_id)
            ]
            res = PostingsList(term=plist.term, doc_frequency=len(filtered_postings))
            res.postings = filtered_postings
            return res

        return plist

    def _read_postings_from_disk(self, offset: int, length: int) -> Optional[PostingsList]:
        if not self.index_dir:
            return None
        postings_path = self.index_dir / "postings.dat"
        if not postings_path.exists():
            return None

        with open(postings_path, "rb") as f:
            f.seek(offset)
            compressed_data = f.read(length)
            raw_json = zlib.decompress(compressed_data).decode("utf-8")
            data = json.loads(raw_json)
            return PostingsList.from_dict(data)

    def save(self, output_dir: str):
        """Save the inverted index to disk in a compact, scalable layout."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.index_dir = out

        postings_path = out / "postings.dat"
        offset = 0

        with open(postings_path, "wb") as pf:
            for term in sorted(self._postings_cache.keys()):
                plist = self._postings_cache[term]
                # Serialize and compress individual postings list
                serialized = json.dumps(plist.to_dict()).encode("utf-8")
                compressed = zlib.compress(serialized, level=6)
                length = len(compressed)

                pf.write(compressed)

                total_tf = sum(p.tf for p in plist.postings)
                self.dictionary.add_or_update(
                    term=term,
                    doc_frequency=plist.doc_frequency,
                    offset=offset,
                    length=length,
                    total_tf=total_tf,
                )
                offset += length

        # Write dictionary and document registry metadata
        meta = {
            "dictionary": self.dictionary.to_dict(),
            "registry": self.document_registry.to_dict(),
        }
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, index_dir: str):
        """Load the index metadata into memory from disk."""
        inp = Path(index_dir)
        self.index_dir = inp
        meta_path = inp / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found in {index_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dictionary = Dictionary.from_dict(meta["dictionary"])
        self.document_registry = DocumentRegistry.from_dict(meta["registry"])
        self._postings_cache.clear()

    def get_doc_name(self, doc_id: int) -> Optional[str]:
        return self.document_registry.get_name(doc_id)

    def get_doc_id(self, doc_name: str) -> Optional[int]:
        return self.document_registry.get_id(doc_name)

    def doc_count(self) -> int:
        return self.document_registry.active_doc_count()

    def vocabulary(self) -> List[str]:
        if self._postings_cache:
            return sorted(set(self.dictionary.vocabulary()).union(self._postings_cache.keys()))
        return self.dictionary.vocabulary()
