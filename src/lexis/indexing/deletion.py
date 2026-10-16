"""Deletion manager supporting logical tombstones and physical index compaction."""

from typing import Optional
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList


class DeletionManager:
    """Manages logical and physical deletion of documents from the index."""

    def __init__(self, index: PositionalIndex):
        self.index = index

    def delete_document(self, doc_name: str) -> bool:
        """Logically delete a document by name using a tombstone.
        
        Returns True if the document was found and marked deleted, False otherwise.
        """
        doc_id = self.index.get_doc_id(doc_name)
        if doc_id is None:
            return False
        self.index.document_registry.mark_deleted(doc_id)
        # Re-save metadata so deleted list persists
        if self.index.index_dir:
            meta = {
                "dictionary": self.index.dictionary.to_dict(),
                "registry": self.index.document_registry.to_dict(),
            }
            import json
            with open(self.index.index_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        return True

    def vacuum(self, output_dir: Optional[str] = None):
        """Physically rebuild and compact the index, purging all tombstoned documents."""
        target_dir = output_dir or str(self.index.index_dir)
        deleted = self.index.document_registry.deleted_doc_ids
        if not deleted:
            return

        active_terms = self.index.dictionary.vocabulary()
        for term in active_terms:
            plist = self.index.get_postings(term)
            if plist:
                filtered_postings = [p for p in plist.postings if p.doc_id not in deleted]
                if filtered_postings:
                    plist.postings = filtered_postings
                    plist.doc_frequency = len(filtered_postings)
                    self.index._postings_cache[term] = plist
                else:
                    # Term has no remaining occurrences
                    if term in self.index._postings_cache:
                        del self.index._postings_cache[term]
                    if self.index.dictionary.contains(term):
                        del self.index.dictionary.terms[term]

        self.index.document_registry.purge_deleted()
        self.index.save(target_dir)
