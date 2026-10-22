import csv
import io

import pytest

from modules.errors import NavigatorError
from modules.experiments import csv_bytes, evaluate, run_benchmark, summarize
from modules.service import Navigator


def test_benchmark_persists_repeats_boundaries_quality_and_cpu_memory(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr('modules.experiments.environment', lambda: {'test_environment': True})
    nav = Navigator(tmp_path / 'data')
    nav.upload('one.txt', 'Точный текст'.encode())
    engine, _ = nav.build(encoder)
    labels = [{'id': 'q1', 'query': 'Точный текст', 'relevant_chunk_ids': [engine.chunks[0]['id']]}]
    rows, folder = run_benchmark(nav.database, encoder, tmp_path / 'results', labels=labels)
    assert len(rows) == 3
    assert encoder.resets == 3
    assert encoder.syncs >= 10
    assert all(r['device'] == 'cpu' and r['peak_allocated_mib'] is None for r in rows)
    assert all(r['hit_at_5'] == 1 and r['status'] == 'ok' for r in rows)
    assert all(r['total_seconds'] >= r['encode_seconds'] > 0 for r in rows)
    assert len(list(csv.DictReader(io.StringIO((folder / 'quality.csv').read_text('utf-8-sig'))))) == 3
    saved = list(csv.DictReader(io.StringIO((folder / 'runs.csv').read_text('utf-8-sig'))))
    summary = summarize(saved)
    assert summary[0]['speedup_total'] is None
    assert summary[0]['peak_allocated_mib'] is None
    assert csv_bytes(rows).startswith(b'\xef\xbb\xbf')
    assert not (folder / 'measured-index.npz').exists()


def test_missing_labels_and_invalid_protocol_rejected(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr('modules.experiments.environment', lambda: {})
    nav = Navigator(tmp_path / 'data')
    nav.upload('one.txt', b'Text')
    engine, _ = nav.build(encoder)
    with pytest.raises(NavigatorError, match='Разметка'):
        evaluate(engine, [{'query': 'Text', 'relevant_chunk_ids': ['absent']}])
    with pytest.raises(NavigatorError, match='минимум три'):
        run_benchmark(nav.database, encoder, tmp_path, repeats=1)


def test_warmup_failure_saved_not_reported_as_success(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr('modules.experiments.environment', lambda: {})
    nav = Navigator(tmp_path / 'data')
    nav.upload('one.txt', b'Text')
    encoder.fail = True
    rows, folder = run_benchmark(nav.database, encoder, tmp_path / 'results')
    assert rows[0]['status'] == 'error' and rows[0]['phase'] == 'warmup'
    assert 'out-of-memory' in (folder / 'runs.csv').read_text('utf-8-sig')
    assert summarize(rows) == []


def test_synthetic_load_labelled_and_quality_cannot_be_mixed(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr('modules.experiments.environment', lambda: {})
    nav = Navigator(tmp_path / 'data')
    nav.upload('one.txt', b'Text')
    rows, _ = run_benchmark(nav.database, encoder, tmp_path / 'results', target_fragments=100)
    assert all(r['synthetic'] and r['n_chunks'] == 100 for r in rows)
    with pytest.raises(NavigatorError, match='дубликаты'):
        run_benchmark(nav.database, encoder, tmp_path, target_fragments=100, labels=[{}])
