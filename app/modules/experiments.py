"""Reproducible indexing measurements, with explicit timing boundaries and raw repeats."""
import csv
import importlib.metadata
import io
import json
import platform
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_CHUNKS
from .documents import canonical, corpus_fingerprint, digest
from .errors import NavigatorError
from .index_store import expected_metadata, save_index
from .search_engine import DocumentSearchEngine
from .text_processing import chunk_documents

PROTOCOL = 'index-v1: db-read+chunk+encode+npz-fsync; excludes model-load,warmup,quality,CSV'


def environment():
    import psutil
    import torch
    info = {'python': platform.python_version(), 'os': platform.platform(),
            'ram_gib': round(psutil.virtual_memory().total / 2**30, 3),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'tf32': False, 'attention': 'eager'}
    for package in ('sentence-transformers', 'transformers', 'numpy', 'streamlit', 'pypdf', 'python-docx'):
        info[package] = importlib.metadata.version(package)
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0),
                    vram_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
                    capability=list(torch.cuda.get_device_capability(0)))
    try:
        info['nvidia_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()
    except (OSError, subprocess.SubprocessError):
        info['nvidia_driver'] = None
    return info


def load_labels(path):
    try:
        labels = json.loads(Path(path).read_text('utf-8'))
        if not isinstance(labels, list) or not labels:
            raise ValueError('Нужен непустой список вопросов.')
        ids = set()
        for label in labels:
            if (not isinstance(label, dict) or not isinstance(label.get('id'), str)
                    or not isinstance(label.get('query'), str) or not label['query'].strip()
                    or not isinstance(label.get('relevant_chunk_ids'), list)
                    or not label['relevant_chunk_ids']
                    or any(not isinstance(i, str) for i in label['relevant_chunk_ids'])
                    or label['id'] in ids):
                raise ValueError('Неверные id, query или relevant_chunk_ids.')
            ids.add(label['id'])
        return labels
    except (OSError, ValueError, TypeError) as exc:
        raise NavigatorError(f'Не удалось прочитать разметку вопросов: {exc}') from exc


def validate_labels(labels, chunks):
    if labels is None:
        return
    ids = {c['id'] for c in chunks}
    missing = {i for label in labels for i in label['relevant_chunk_ids']} - ids
    if missing:
        raise NavigatorError('Разметка не соответствует фрагментам коллекции: ' + ', '.join(sorted(missing)))


def evaluate(engine, labels):
    validate_labels(labels, engine.chunks)
    answers = []
    for label in labels:
        engine.encoder.sync()
        start = time.perf_counter()
        results = engine.search(label['query'])
        engine.encoder.sync()
        elapsed = time.perf_counter() - start
        found = [r.item['id'] for r in results]
        answers.append({**label, 'retrieved_chunk_ids': found, 'scores': [r.score for r in results],
                        'hit_at_5': int(bool(set(found) & set(label['relevant_chunk_ids']))),
                        'query_seconds': elapsed})
    return statistics.mean(a['hit_at_5'] for a in answers), answers


def csv_bytes(rows):
    if not rows:
        return b''
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def _load_corpus(database, encoder, config, target_fragments):
    documents = database.get_documents()
    chunks = chunk_documents(documents, encoder.tokenizer, config)
    if target_fragments is not None:
        if target_fragments < len(chunks):
            raise NavigatorError('Искусственная нагрузка должна быть не меньше исходной коллекции.')
        original = list(chunks)
        for i in range(len(chunks), target_fragments):
            chunk = original[i % len(original)]
            chunks.append({**chunk, 'id': f'{chunk["id"]}:synthetic:{i}'})
    return documents, chunks


def run_benchmark(database, encoder, output_dir, *, batches=(32,), repeats=3,
                  warmups=1, config=DEFAULT_CHUNKS, labels=None, target_fragments=None,
                  progress=None):
    if repeats < 3 or warmups < 1 or not batches or any(b not in (8, 16, 32, 64, 128) for b in batches):
        raise NavigatorError('Нужны прогрев, минимум три повтора и пакеты 8/16/32/64/128.')
    if labels and target_fragments:
        raise NavigatorError('Оцените качество отдельно на исходной коллекции: дубликаты искажают Hit@5.')
    documents, prepared = _load_corpus(database, encoder, config, target_fragments)
    validate_labels(labels, prepared)
    fingerprint = corpus_fingerprint(documents)
    chunks_hash = digest(canonical(prepared).encode('utf-8'))
    env = environment()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = stamp + '-' + uuid.uuid4().hex[:8]
    folder = Path(output_dir) / run_id
    folder.mkdir(parents=True)
    key_data = {'corpus': fingerprint, 'chunks': chunks_hash, 'model': encoder.signature,
                'chunk_config': config.to_dict(), 'environment': env, 'protocol': PROTOCOL,
                'labels_sha256': digest(canonical(labels).encode()) if labels else None}
    common = {'run_id': run_id, 'created_at': stamp, 'comparison_key': digest(canonical(key_data).encode()),
              'device': encoder.device, 'precision': encoder.precision, 'n_chunks': len(prepared),
              'synthetic': target_fragments is not None, 'source_documents': len(documents),
              'corpus_sha256': fingerprint, 'chunks_sha256': chunks_hash,
              'model': encoder.signature, 'chunk_config': config.to_dict(), 'environment': env,
              'protocol': PROTOCOL, 'warmups': warmups, 'repeats': repeats,
              'labels_sha256': key_data['labels_sha256'], 'labels_count': len(labels) if labels else 0}
    (folder / 'manifest.json').write_text(canonical(common), encoding='utf-8')
    (folder / 'chunks.json').write_text(canonical(prepared), encoding='utf-8')
    if labels:
        (folder / 'labels.json').write_text(canonical(labels), encoding='utf-8')
    rows, answers = [], []

    def persist():
        (folder / 'runs.csv').write_bytes(csv_bytes(rows))
        (folder / 'quality.csv').write_bytes(csv_bytes(answers))

    for batch in batches:
        try:
            for _ in range(warmups):
                encoder.encode([c['text'] for c in prepared], batch_size=batch)
                encoder.sync()
            if labels:
                encoder.encode([labels[0]['query']], query=True, batch_size=1)
                encoder.sync()
        except Exception as exc:
            rows.append({**common, 'batch_size': batch, 'repeat': 0, 'status': 'error',
                         'phase': 'warmup', 'error': str(exc)})
            persist()
            continue
        for repeat in range(1, repeats + 1):
            row = {**common, 'batch_size': batch, 'repeat': repeat, 'status': 'ok', 'phase': 'index', 'error': ''}
            engine = None
            temp_index = folder / 'measured-index.npz'
            try:
                encoder.sync()
                encoder.reset_peak_memory()
                start = time.perf_counter()
                docs_now, chunks = _load_corpus(database, encoder, config, target_fragments)
                preparation_end = time.perf_counter()
                engine = DocumentSearchEngine(encoder)
                engine.build(chunks, batch)
                encoder.sync()
                neural_end = time.perf_counter()
                # Fingerprint validation and index serialization are included in the full boundary.
                if corpus_fingerprint(docs_now) != fingerprint or digest(canonical(chunks).encode()) != chunks_hash:
                    raise NavigatorError('Коллекция изменилась во время измерений. Повторите эксперимент.')
                save_index(temp_index, chunks, engine.embeddings,
                           expected_metadata(fingerprint, encoder.signature, config),
                           device=encoder.device, precision=encoder.precision, batch_size=batch)
                end = time.perf_counter()
                row.update(prepare_seconds=preparation_end - start, encode_seconds=neural_end - preparation_end,
                           save_seconds=end - neural_end, total_seconds=end - start,
                           chunks_per_second=len(chunks) / (neural_end - preparation_end),
                           **encoder.memory())
                if labels:
                    row['phase'] = 'quality'
                    row['hit_at_5'], details = evaluate(engine, labels)
                    row['query_median_seconds'] = statistics.median(d['query_seconds'] for d in details)
                    answers.extend({**d, 'run_id': run_id, 'batch_size': batch, 'repeat': repeat,
                                    'device': encoder.device, 'precision': encoder.precision} for d in details)
                row['phase'] = 'complete'
            except Exception as exc:
                row.update(status='error', error=str(exc))
            finally:
                engine = None
                temp_index.unlink(missing_ok=True)
            rows.append(row)
            persist()
            if progress:
                progress(f'{encoder.device.upper()} {encoder.precision.upper()}, пакет {batch}, повтор {repeat}/{repeats}: {row["status"]}')
    return rows, folder


def summarize(rows):
    groups = {}
    for row in rows:
        if row['status'] == 'ok':
            key = (row['run_id'], row['comparison_key'], row['device'], row['precision'], int(row['batch_size']))
            groups.setdefault(key, []).append(row)
    summary = []
    for (run, compare, device, precision, batch), group in groups.items():
        row = dict(run_id=run, comparison_key=compare, device=device, precision=precision, batch_size=batch,
                   successful_repeats=len(group), n_chunks=int(group[0]['n_chunks']))
        for field in ('total_seconds', 'encode_seconds', 'prepare_seconds', 'save_seconds', 'chunks_per_second',
                      'peak_allocated_mib', 'peak_reserved_mib', 'hit_at_5', 'query_median_seconds'):
            values = [float(r[field]) for r in group if r.get(field) not in (None, '')]
            row[field] = statistics.median(values) if values else None
        row['total_min_seconds'] = min(float(r['total_seconds']) for r in group)
        row['total_max_seconds'] = max(float(r['total_seconds']) for r in group)
        summary.append(row)
    for row in summary:
        bases = [r for r in summary if r['comparison_key'] == row['comparison_key']
                 and r['device'] == 'cuda' and r['precision'] == 'fp32' and r['batch_size'] == 32
                 and r['successful_repeats'] >= 3]
        # Most recent baseline with identical data/model/environment/labels/protocol.
        base = sorted(bases, key=lambda r: r['run_id'])[-1] if bases else None
        row['baseline_run_id'] = base['run_id'] if base else None
        eligible = base and row['device'] == 'cuda' and row['successful_repeats'] >= 3
        row['speedup_total'] = base['total_seconds'] / row['total_seconds'] if eligible else None
        row['quality_not_lower'] = (row['hit_at_5'] >= base['hit_at_5']
                                   if eligible and row['hit_at_5'] is not None and base['hit_at_5'] is not None else None)
    return summary
