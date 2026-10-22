import hashlib
import re

import numpy as np
import pytest


class WordTokenizer:
    """A test double; its vectors are never used as experiment evidence."""
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {'length': [len(t.split()) + 2 for t in text]}
        offsets = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        return {'offset_mapping': offsets, 'input_ids': list(range(len(offsets) + 2))}


class FakeEncoder:
    device = 'cpu'
    precision = 'fp32'
    signature = {'model_id': 'test-only', 'revision': 'test-v1', 'embedding_dim': 16}

    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.calls = []
        self.syncs = 0
        self.resets = 0
        self.fail = False

    def encode(self, texts, *, query=False, batch_size=32):
        self.calls.append((list(texts), query, batch_size))
        if self.fail:
            raise RuntimeError('simulated out-of-memory')
        result = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=16).astype(np.float32)
            result.append(vector / np.linalg.norm(vector))
        return np.array(result)

    def sync(self):
        self.syncs += 1

    def reset_peak_memory(self):
        self.resets += 1

    def memory(self):
        return {'peak_allocated_mib': None, 'peak_reserved_mib': None}


@pytest.fixture
def encoder():
    return FakeEncoder()
