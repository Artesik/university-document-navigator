import json

import numpy as np
import pytest

from modules.config import ChunkConfig
from modules.errors import IndexError, NavigatorError
from modules.service import Navigator


def test_saved_index_reopens_without_document_encoding_and_search_has_source(tmp_path, encoder):
    nav = Navigator(tmp_path)
    for i in range(8):
        nav.upload(f'test{i}.txt', f'Документ {i}'.encode())
    engine, metadata = nav.build(encoder)
    before = len(encoder.calls)
    reopened, loaded = Navigator(tmp_path).open(encoder)
    assert len(encoder.calls) == before
    assert metadata == loaded
    assert np.array_equal(engine.embeddings, reopened.embeddings)
    results = reopened.search('Документ 3')
    assert len(results) == 5
    assert results[0].item['document_name'] == 'test3.txt'
    assert results[0].item['number'] == 1
    assert results[0].score == pytest.approx(1, abs=1e-5)
    assert encoder.calls[-1][1] is True


def test_index_detects_data_model_and_chunk_changes(tmp_path, encoder):
    nav = Navigator(tmp_path)
    nav.upload('one.txt', b'One document')
    nav.build(encoder)
    with pytest.raises(IndexError, match='устарел'):
        nav.open(encoder, ChunkConfig(256, 32))
    encoder.signature = {**encoder.signature, 'revision': 'different'}
    with pytest.raises(IndexError, match='устарел'):
        nav.open(encoder)
    encoder.signature = {**encoder.signature, 'revision': 'test-v1'}
    nav.upload('two.txt', b'Two documents')
    with pytest.raises(IndexError, match='устарел'):
        nav.open(encoder)


def test_corrupt_index_and_failed_rebuild_preserve_previous_snapshot(tmp_path, encoder):
    nav = Navigator(tmp_path)
    with pytest.raises(IndexError, match='ещё не построен'):
        nav.open(encoder)
    nav.upload('test.txt', b'Test text')
    nav.build(encoder)
    previous = nav.index_path.read_bytes()
    encoder.fail = True
    with pytest.raises(RuntimeError):
        nav.build(encoder)
    assert nav.index_path.read_bytes() == previous
    encoder.fail = False
    # A valid ZIP with tampered JSON must also be rejected (not only broken ZIPs).
    with np.load(nav.index_path, allow_pickle=False) as data:
        chunks = json.loads(data['chunks'].item())
        chunks[0]['document_name'] = 'incorrect source'
        vectors, metadata = data['vectors'], data['metadata']
    np.savez(nav.index_path, vectors=vectors, chunks=np.array(json.dumps(chunks)), metadata=metadata)
    with pytest.raises(IndexError, match='целостность'):
        nav.open(encoder)
    nav.index_path.write_bytes(b'broken')
    with pytest.raises(IndexError, match='повреждён'):
        nav.open(encoder)


def test_failed_atomic_replace_does_not_lose_old_index(tmp_path, encoder, monkeypatch):
    nav = Navigator(tmp_path)
    nav.upload('test.txt', b'Test text')
    nav.build(encoder)
    previous = nav.index_path.read_bytes()
    def fail_replace(*args):
        raise PermissionError('file in use')
    monkeypatch.setattr('modules.index_store.os.replace', fail_replace)
    with pytest.raises(IndexError, match='сохранить'):
        nav.build(encoder)
    assert nav.index_path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))


def test_empty_query_empty_collection_and_missing_index(tmp_path, encoder):
    nav = Navigator(tmp_path)
    with pytest.raises(NavigatorError, match='Нет фрагментов'):
        nav.build(encoder)
    nav.upload('test.txt', b'Test')
    engine, _ = nav.build(encoder)
    assert len(engine.search('Test')) == 1
    with pytest.raises(NavigatorError, match='непустой'):
        engine.search(' \t')
    with pytest.raises(NavigatorError, match='пяти'):
        engine.search('Test', top_k=6)
