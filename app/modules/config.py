"""Versioned processing defaults; runtime paths are independent of the working directory."""
import os
from dataclasses import asdict, dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('NAVIGATOR_DATA_DIR', str(APP_DIR / 'data'))).resolve()
MODEL_DIR = APP_DIR / 'models' / 'multilingual-e5-small'
MODEL_ID = 'intfloat/multilingual-e5-small'
BASELINE_BATCH = 32
BATCH_SIZES = (8, 16, 32, 64, 128)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ChunkConfig:
    tokens: int = 384
    overlap: int = 64
    version: str = 'e5-offsets-v1'

    def __post_init__(self):
        if not 32 <= self.tokens <= 480 or not 0 <= self.overlap < self.tokens:
            raise ValueError('Размер фрагмента: 32–480 токенов; перекрытие меньше размера.')

    def to_dict(self):
        return asdict(self)


DEFAULT_CHUNKS = ChunkConfig()
