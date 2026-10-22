"""Russian text normalization helpers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[а-яa-z0-9]+", re.IGNORECASE)
_STOP_WORDS = {
    "а", "без", "бы", "в", "во", "вы", "для", "до", "его", "ее", "если",
    "же", "за", "и", "из", "или", "к", "как", "ко", "мне", "мой", "мы",
    "на", "над", "не", "но", "о", "об", "он", "она", "они", "от", "по",
    "под", "при", "с", "со", "то", "у", "что", "это", "я",
}


def normalize_text(text: str) -> str:
    """Normalize casing, ё and whitespace while preserving useful symbols."""
    return " ".join(text.lower().replace("ё", "е").strip().split())


def tokenize(text: str) -> list[str]:
    """Return meaningful lowercase tokens for the lexical fallback search."""
    return [token for token in _TOKEN_RE.findall(normalize_text(text)) if token not in _STOP_WORDS]


def char_ngrams(text: str, size: int = 3) -> set[str]:
    """Build character n-grams that tolerate Russian word-form changes."""
    compact = normalize_text(text).replace(" ", "_")
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def chunk_documents(documents, tokenizer, config):
    """Token windows over original character spans; PDF pages never get merged."""
    from .errors import DocumentError
    chunks = []
    for doc in sorted(documents, key=lambda d: d['id']):
        number = 0
        for page in doc['pages']:
            text = page['text']
            offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)['offset_mapping']
            start = 0
            while start < len(offsets):
                end = min(start + config.tokens, len(offsets))
                left, right = offsets[start][0], offsets[end - 1][1]
                fragment = text[left:right].strip()
                # Re-tokenization at a substring boundary may alter token count.
                while len(tokenizer('passage: ' + fragment)['input_ids']) > 512 and end > start + 1:
                    end -= 1
                    right = offsets[end - 1][1]
                    fragment = text[left:right].strip()
                if fragment:
                    number += 1
                    chunks.append({'id': f'{doc["id"]}:p{page["page"] or 0}:c{number}',
                                   'document_id': doc['id'], 'document_name': doc['name'],
                                   'page': page['page'], 'number': number, 'text': fragment,
                                   'char_start': left, 'char_end': right, 'metadata': doc['metadata']})
                if end == len(offsets):
                    break
                start = max(start + 1, end - config.overlap)
    if not chunks:
        raise DocumentError('Нет фрагментов. Загрузите документ с текстом или импортируйте инструкции.')
    return chunks
