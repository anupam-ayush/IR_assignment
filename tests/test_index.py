"""Unit tests for index construction, skip pointers, incremental updates, and deletions."""

import shutil
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.index.skip_pointers import intersect_with_skips, intersect_without_skips
from lexis.indexing.incremental import IncrementalIndexer
from lexis.indexing.deletion import DeletionManager
from lexis.indexing.document_reader import Document
from lexis.preprocessing.tokenizer import Tokenizer


def test_index_save_and_load():
    tmp_dir = tempfile.mkdtemp()
    try:
        index = PositionalIndex(tmp_dir)
        reg = index.document_registry
        did0 = reg.register("doc_01", 10, 2, 8)
        did1 = reg.register("doc_02", 15, 3, 12)

        index.add_term_occurrence("gold", did0, 0, "title")
        index.add_term_occurrence("silver", did0, 1, "title")
        index.add_term_occurrence("gold", did1, 2, "body")

        index.save(tmp_dir)

        # Reload
        loaded = PositionalIndex(tmp_dir)
        loaded.load(tmp_dir)

        assert loaded.doc_count() == 2
        assert loaded.dictionary.contains("gold")
        assert loaded.dictionary.contains("silver")

        plist_gold = loaded.get_postings("gold")
        assert plist_gold.doc_frequency == 2
        assert plist_gold.doc_ids() == [did0, did1]

        plist_silver = loaded.get_postings("silver")
        assert plist_silver.doc_frequency == 1
        assert plist_silver.doc_ids() == [did0]
    finally:
        shutil.rmtree(tmp_dir)


def test_skip_pointers_intersection():
    p1 = list(range(1, 21))
    p2 = [4, 8, 12, 16, 20]

    # Without skips
    res_no_skip, stats_no_skip = intersect_without_skips(p1, p2)
    assert res_no_skip == [4, 8, 12, 16, 20]

    # With skips
    res_skip, stats_skip = intersect_with_skips(p1, p2, spacing1=3, spacing2=3)
    assert res_skip == [4, 8, 12, 16, 20]
    assert stats_skip.skips_taken >= 1


def test_incremental_and_deletion():
    tmp_dir = tempfile.mkdtemp()
    try:
        index = PositionalIndex(tmp_dir)
        tok = Tokenizer()
        inc = IncrementalIndexer(index, tok)

        doc1 = Document(doc_name="doc_01", title="US Economy", body="Inflation remains low.")
        doc2 = Document(doc_name="doc_02", title="Japan Trade", body="Surplus increases.")

        inc.add_document(doc1)
        inc.add_document(doc2)
        inc.commit(tmp_dir)

        assert index.doc_count() == 2
        assert index.get_postings("economy") is not None

        # Logical deletion
        del_mgr = DeletionManager(index)
        assert del_mgr.delete_document("doc_01")
        assert index.doc_count() == 1
        assert index.get_postings("economy").doc_ids() == []

        # Physical vacuum
        del_mgr.vacuum(tmp_dir)
        assert index.doc_count() == 1
    finally:
        shutil.rmtree(tmp_dir)
