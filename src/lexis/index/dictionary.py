"""Term dictionary and document registry for Lexis index."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TermInfo:
    term: str
    doc_frequency: int
    offset: int = 0
    length: int = 0
    total_tf: int = 0


class DocumentRegistry:
    """Bi-directional mapping between integer doc_ids and document names (e.g. 'doc_00001')."""

    def __init__(self):
        self.doc_id_to_name: Dict[int, str] = {}
        self.name_to_doc_id: Dict[str, int] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.title_lengths: Dict[int, int] = {}
        self.body_lengths: Dict[int, int] = {}
        self.deleted_doc_ids: set[int] = set()

    def register(self, doc_name: str, doc_length: int = 0, title_length: int = 0, body_length: int = 0) -> int:
        if doc_name in self.name_to_doc_id:
            int_id = self.name_to_doc_id[doc_name]
            self.deleted_doc_ids.discard(int_id)
        else:
            int_id = len(self.doc_id_to_name)
            self.doc_id_to_name[int_id] = doc_name
            self.name_to_doc_id[doc_name] = int_id

        self.doc_lengths[int_id] = doc_length
        self.title_lengths[int_id] = title_length
        self.body_lengths[int_id] = body_length
        return int_id

    def get_name(self, doc_id: int) -> Optional[str]:
        return self.doc_id_to_name.get(doc_id)

    def get_id(self, doc_name: str) -> Optional[int]:
        return self.name_to_doc_id.get(doc_name)

    def mark_deleted(self, doc_id: int):
        self.deleted_doc_ids.add(doc_id)

    def is_deleted(self, doc_id: int) -> bool:
        return doc_id in self.deleted_doc_ids

    def active_doc_count(self) -> int:
        return len(self.doc_id_to_name) - len(self.deleted_doc_ids)

    def total_doc_count(self) -> int:
        return len(self.doc_id_to_name)

    def purge_deleted(self):
        """Physically remove deleted documents from registry mappings."""
        for doc_id in list(self.deleted_doc_ids):
            name = self.doc_id_to_name.pop(doc_id, None)
            if name and name in self.name_to_doc_id:
                del self.name_to_doc_id[name]
            self.doc_lengths.pop(doc_id, None)
            self.title_lengths.pop(doc_id, None)
            self.body_lengths.pop(doc_id, None)
        self.deleted_doc_ids.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id_to_name": {str(k): v for k, v in self.doc_id_to_name.items()},
            "name_to_doc_id": self.name_to_doc_id,
            "doc_lengths": {str(k): v for k, v in self.doc_lengths.items()},
            "title_lengths": {str(k): v for k, v in self.title_lengths.items()},
            "body_lengths": {str(k): v for k, v in self.body_lengths.items()},
            "deleted_doc_ids": list(self.deleted_doc_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRegistry":
        reg = cls()
        reg.doc_id_to_name = {int(k): v for k, v in data.get("doc_id_to_name", {}).items()}
        reg.name_to_doc_id = data.get("name_to_doc_id", {})
        reg.doc_lengths = {int(k): v for k, v in data.get("doc_lengths", {}).items()}
        reg.title_lengths = {int(k): v for k, v in data.get("title_lengths", {}).items()}
        reg.body_lengths = {int(k): v for k, v in data.get("body_lengths", {}).items()}
        reg.deleted_doc_ids = set(data.get("deleted_doc_ids", []))
        return reg


class Dictionary:
    """Vocabulary dictionary storing metadata and postings offsets for all terms."""

    def __init__(self):
        self.terms: Dict[str, TermInfo] = {}

    def add_or_update(self, term: str, doc_frequency: int, offset: int = 0, length: int = 0, total_tf: int = 0):
        self.terms[term] = TermInfo(
            term=term,
            doc_frequency=doc_frequency,
            offset=offset,
            length=length,
            total_tf=total_tf,
        )

    def contains(self, term: str) -> bool:
        return term in self.terms

    def get(self, term: str) -> Optional[TermInfo]:
        return self.terms.get(term)

    def vocabulary(self) -> List[str]:
        return sorted(self.terms.keys())

    def size(self) -> int:
        return len(self.terms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            term: {
                "df": info.doc_frequency,
                "offset": info.offset,
                "length": info.length,
                "total_tf": info.total_tf,
            }
            for term, info in self.terms.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dictionary":
        d = cls()
        for term, info in data.items():
            d.add_or_update(
                term=term,
                doc_frequency=info["df"],
                offset=info.get("offset", 0),
                length=info.get("length", 0),
                total_tf=info.get("total_tf", 0),
            )
        return d
