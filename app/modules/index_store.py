"""One atomic NPZ snapshot: no pickle; metadata, dimensions and digests are checked."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .documents import canonical, digest
from .errors import IndexError

FORMAT_VERSION = 1


def expected_metadata(fingerprint, signature, chunk_config):
    return {'format_version': FORMAT_VERSION, 'corpus_sha256': fingerprint,
            'model': signature, 'chunk_config': chunk_config.to_dict()}


def save_index(path, chunks, vectors, expected, *, device, precision, batch_size):
    path = Path(path)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    _validate(chunks, vectors, expected['model']['embedding_dim'])
    chunks_json = canonical(chunks)
    metadata = {**expected, 'created_at': datetime.now(timezone.utc).isoformat(),
                'device': device, 'precision': precision, 'batch_size': batch_size,
                'chunks_sha256': digest(chunks_json.encode('utf-8')),
                'vectors_sha256': digest(vectors.tobytes())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez(stream, vectors=vectors, chunks=np.array(chunks_json), metadata=np.array(canonical(metadata)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        raise IndexError(f'Не удалось сохранить индекс: {exc}. Предыдущий файл сохранён, если запись не завершилась.') from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
    return metadata


def _validate(chunks, vectors, dim):
    if not isinstance(chunks, list) or not chunks or vectors.shape != (len(chunks), dim):
        raise ValueError('Неверная размерность индекса или отсутствуют фрагменты.')
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError('Некорректные значения векторов.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError('Векторы индекса не нормализованы.')
    ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or not all(k in chunk for k in
                ('id', 'document_id', 'document_name', 'page', 'number', 'text', 'metadata')):
            raise ValueError('Утеряна привязка фрагмента к источнику.')
        if chunk['id'] in ids or not chunk['text'].strip() or chunk['number'] < 1:
            raise ValueError('Некорректные идентификаторы или текст фрагментов.')
        ids.add(chunk['id'])


def load_index(path, expected):
    if not Path(path).exists():
        raise IndexError('Индекс ещё не построен. Откройте вкладку «Документы».')
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            chunks_json = str(archive['chunks'].item())
            chunks = json.loads(chunks_json)
            vectors = archive['vectors'].copy()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise IndexError('Индекс устарел: изменились документы, модель или параметры фрагментации. Перестройте индекс.')
        if metadata.get('chunks_sha256') != digest(chunks_json.encode('utf-8')):
            raise ValueError('Нарушена целостность метаданных фрагментов.')
        if metadata.get('vectors_sha256') != digest(vectors.tobytes()):
            raise ValueError('Нарушена целостность векторов.')
        _validate(chunks, vectors, expected['model']['embedding_dim'])
        return chunks, vectors, metadata
    except IndexError:
        raise
    except Exception as exc:
        raise IndexError(f'Не удалось открыть индекс: файл повреждён или несовместим. Перестройте его. Подробности: {exc}') from exc
