"""Opt-in integration checks; real CUDA only, never counted as an optimization series."""
import os
import socket
from pathlib import Path

import pytest

from modules.config import ChunkConfig
from modules.embedding import E5Encoder
from modules.errors import ModelError
from modules.experiments import evaluate, load_labels
from modules.service import Navigator
from modules.text_processing import chunk_documents

pytestmark = pytest.mark.real_model


@pytest.fixture(scope='module')
def real_encoder():
    if os.environ.get('NAVIGATOR_REAL_GPU') != '1':
        pytest.skip('Нужны локальная E5 и CUDA; для реальной проверки установите NAVIGATOR_REAL_GPU=1.')
    import torch
    assert torch.cuda.is_available(), 'Запрошена GPU-проверка, но CUDA недоступна'
    return E5Encoder('cuda', 'fp32')


def test_gpu_offline_import_three_formats_reopen_quality(tmp_path, real_encoder, monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError('Network request attempted during local inference')
    monkeypatch.setattr(socket.socket, 'connect', offline)
    nav = Navigator(tmp_path)
    nav.database.import_demo_documents('database/knowledge_base.json')
    for path in (Path('examples/library.txt'), Path('examples/practice.docx'), Path('examples/document_rules.pdf')):
        nav.upload(path.name, path.read_bytes())
    engine, _ = nav.build(real_encoder)
    hit, answers = evaluate(engine, load_labels('examples/quality_questions.json'))
    assert hit >= 0.8
    assert all(len(a['retrieved_chunk_ids']) == 5 for a in answers)
    results = engine.search('Где найти учебники в электронной библиотеке?')
    assert any(r.item['document_name'] == 'library.txt' for r in results)
    reopened, _ = Navigator(tmp_path).open(real_encoder)
    assert reopened.search('Где найти учебники в электронной библиотеке?')[0].item == results[0].item
    pdf = reopened.search('Как проверить номер версии регламента?')
    assert any(r.item['document_name'] == 'document_rules.pdf' and r.item['page'] == 2 for r in pdf)


def test_real_token_boundaries_and_long_query_error(tmp_path, real_encoder):
    nav = Navigator(tmp_path)
    nav.upload('long.txt', ('Ёжик, университет: эксперимент! ' * 500).encode())
    chunks = chunk_documents(nav.database.get_documents(), real_encoder.tokenizer, ChunkConfig())
    assert len(chunks) > 2
    assert all(len(real_encoder.tokenizer('passage: ' + c['text'])['input_ids']) <= 512 for c in chunks)
    with pytest.raises(ModelError, match='512'):
        real_encoder.encode(['университет ' * 1000], query=True)


def test_invalid_devices_do_not_fallback():
    with pytest.raises(ModelError, match='только для CUDA'):
        E5Encoder('cpu', 'fp16')
