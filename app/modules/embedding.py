"""Explicit local-only E5 inference. GPU failures never fall back to CPU silently."""
import json
from pathlib import Path

import numpy as np

from .config import APP_DIR, MODEL_DIR, MODEL_ID
from .documents import canonical, digest
from .errors import ModelError


class E5Encoder:
    def __init__(self, device='cuda', precision='fp32', model_dir=MODEL_DIR):
        if device not in ('cuda', 'cpu') or precision not in ('fp32', 'fp16'):
            raise ModelError('Допустимые режимы: CUDA/CPU, FP32/FP16.')
        if device == 'cpu' and precision == 'fp16':
            raise ModelError('FP16 предусмотрен только для CUDA. На CPU выберите FP32.')
        self.device = device
        self.precision = precision
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            self.torch = torch
            if device == 'cuda' and not torch.cuda.is_available():
                raise ModelError('CUDA недоступна. Проверьте драйвер и сборку PyTorch либо явно выберите CPU.')
            model_dir = Path(model_dir)
            manifest = json.loads((model_dir / 'model-manifest.json').read_text('utf-8'))
            lock = json.loads((APP_DIR / 'model-lock.json').read_text('utf-8'))
            if manifest['model_id'] != MODEL_ID or manifest['revision'] != lock['revision']:
                raise ModelError('Версия локальной модели отличается от model-lock.json. Повторите скачивание.')
            for name, checksum in manifest['files'].items():
                if digest((model_dir / name).read_bytes()) != checksum:
                    raise ModelError('Файлы модели повреждены. Запустите scripts/download_model.py повторно.')
            self.signature = {**lock, 'files_sha256': digest(canonical(manifest['files']).encode()),
                              'max_seq_length': 512, 'query_prefix': 'query: ', 'passage_prefix': 'passage: ',
                              'normalize': True, 'embedding_dim': 384}
            # Fix FP32 math policy across the baseline and future comparisons.
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.model = SentenceTransformer(str(model_dir), device=device, local_files_only=True,
                                             trust_remote_code=False,
                                             model_kwargs={'attn_implementation': 'eager'})
            self.model.max_seq_length = 512
            self.model.half() if precision == 'fp16' else self.model.float()
            self.model.eval()
            self.tokenizer = self.model.tokenizer
            if not self.tokenizer.is_fast:
                raise ModelError('Для точного разбиения нужен быстрый токенизатор с позициями символов.')
        except ModelError:
            raise
        except Exception as exc:
            if 'torch' in locals() and isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                raise ModelError('Не хватает видеопамяти для загрузки E5. Освободите GPU и повторите попытку. '
                                 'Автоматического перехода на CPU нет.') from exc
            raise ModelError('Не удалось открыть локальную E5-модель. Установите зависимости и выполните '
                             f'python scripts/download_model.py. Подробности: {exc}') from exc

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def reset_peak_memory(self):
        if self.device == 'cuda':
            self.torch.cuda.reset_peak_memory_stats()

    def memory(self):
        if self.device != 'cuda':
            return {'peak_allocated_mib': None, 'peak_reserved_mib': None}
        return {'peak_allocated_mib': self.torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': self.torch.cuda.max_memory_reserved() / 2**20}

    def encode(self, texts, *, query=False, batch_size=32):
        if not texts:
            raise ModelError('Нет текстов для нейросетевой обработки.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ModelError('Размер пакета должен быть положительным целым числом.')
        prefix = 'query: ' if query else 'passage: '
        prepared = [prefix + t.strip() for t in texts]
        lengths = self.tokenizer(prepared, add_special_tokens=True, truncation=False,
                                 return_length=True)['length']
        if max(lengths) > 512:
            raise ModelError('Текст превышает 512 токенов E5. Сократите запрос или размер фрагмента.')
        try:
            with self.torch.inference_mode():
                vectors = self.model.encode(prepared, batch_size=batch_size, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False)
            result = np.asarray(vectors, dtype=np.float32)
            if not np.isfinite(result).all():
                raise ModelError('Модель вернула некорректные числа. Попробуйте FP32.')
            # Dot-product cosine and persisted vectors use FP32 for either inference precision.
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ModelError('Модель вернула нулевой вектор.')
            return result / norms
        except self.torch.cuda.OutOfMemoryError as exc:
            self.torch.cuda.empty_cache()
            raise ModelError('Не хватает видеопамяти CUDA. Уменьшите пакет или закройте другие GPU-программы. '
                             'Режим автоматически не изменён.') from exc
        except ModelError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise ModelError(f'Ошибка нейросетевой обработки: {exc}') from exc
