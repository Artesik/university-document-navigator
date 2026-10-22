from types import SimpleNamespace

import pytest
import torch

from modules.embedding import E5Encoder
from modules.errors import ModelError


def test_unavailable_cuda_is_reported_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ModelError, match='CUDA недоступна'):
        E5Encoder('cuda', 'fp32')


def test_missing_model_explains_local_download_step(tmp_path):
    with pytest.raises(ModelError, match='download_model.py'):
        E5Encoder('cpu', 'fp32', model_dir=tmp_path)


def test_encoding_oom_is_explained_and_does_not_change_device(monkeypatch):
    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError('simulated CUDA allocation failure')
    encoder = E5Encoder.__new__(E5Encoder)
    encoder.device, encoder.precision, encoder.torch = 'cuda', 'fp32', torch
    encoder.tokenizer = lambda *args, **kwargs: {'length': [8]}
    encoder.model = SimpleNamespace(encode=fail)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    with pytest.raises(ModelError, match='Не хватает видеопамяти CUDA'):
        encoder.encode(['Проверка ошибки'])
    assert encoder.device == 'cuda' and encoder.precision == 'fp32'
