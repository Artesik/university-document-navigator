"""Local import of text documents and the original demo knowledge base."""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from .config import MAX_UPLOAD_BYTES
from .errors import DocumentError


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_knowledge(path):
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        items = payload.get('items') if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not items:
            raise ValueError('Ожидается непустой список items.')
        ids = set()
        for item in items:
            if not isinstance(item, dict) or type(item.get('id')) is not int or item['id'] in ids:
                raise ValueError('У инструкций должны быть уникальные целые id.')
            ids.add(item['id'])
            for key in ('question', 'answer', 'category', 'source', 'department'):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f'У инструкции {item["id"]} отсутствует поле {key}.')
            for key in ('aliases', 'steps', 'keywords'):
                if not isinstance(item.get(key, []), list) or any(not isinstance(s, str) for s in item.get(key, [])):
                    raise ValueError(f'Поле {key} должно содержать список строк.')
        return items
    except (OSError, ValueError, TypeError) as exc:
        raise DocumentError(f'Не удалось импортировать демонстрационные инструкции: {exc}') from exc


def instruction_text(item):
    """Expanded legacy _document_text: answers and ordered steps are searchable."""
    return '\n'.join([
        item['question'], ' '.join(item.get('aliases', [])), item['answer'],
        *[f'{i}. {s}' for i, s in enumerate(item.get('steps', []), 1)],
        ' '.join(item.get('keywords', [])), item['category'],
    ])


def demo_document(item):
    raw = canonical(item).encode('utf-8')
    return dict(id=f'legacy-{item["id"]:03d}', name=f'Инструкция {item["id"]}: {item["question"]}',
                format='demo', sha256=digest(raw), metadata={**item, 'demo': True},
                pages=[{'page': None, 'text': instruction_text(item)}], raw=raw)


def _clean(text):
    # Keep case, punctuation, paragraph boundaries and ё for the neural model.
    return text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n').strip()


def extract_document(filename: str, raw: bytes):
    name = re.split(r'[/\\]', filename)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in ('.txt', '.docx', '.pdf'):
        raise DocumentError('Поддерживаются только TXT, DOCX и PDF с текстовым слоем.')
    if not raw:
        raise DocumentError(f'Файл «{name}» пуст.')
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DocumentError('Размер файла превышает 25 МБ.')
    warnings = []
    try:
        if suffix == '.txt':
            encoding = 'utf-8-sig'
            if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
                encoding = 'utf-16'
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                encoding = 'cp1251'
                text = raw.decode(encoding)
            if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                raise ValueError('TXT содержит двоичные данные.')
            pages = [{'page': None, 'text': _clean(text)}]
            warnings.append(f'Кодировка TXT: {encoding}.')
        elif suffix == '.docx':
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(e.file_size for e in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError('Распакованный DOCX превышает 100 МБ.')
            doc = Document(io.BytesIO(raw))
            blocks = []
            for block in doc.iter_inner_content():
                if isinstance(block, Paragraph):
                    blocks.append(block.text)
                elif isinstance(block, Table):
                    blocks.extend(' | '.join(c.text for c in row.cells) for row in block.rows)
            pages = [{'page': None, 'text': _clean('\n'.join(blocks))}]
            warnings.append('DOCX: основной текст и таблицы; номера страниц не назначаются.')
        else:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError('PDF защищён паролем. Загрузите незашифрованную копию.')
            if len(reader.pages) > 1000:
                raise ValueError('PDF содержит более 1000 страниц.')
            pages = [{'page': i, 'text': _clean(page.extract_text() or '')}
                     for i, page in enumerate(reader.pages, 1)]
            blank = [p['page'] for p in pages if not p['text']]
            if blank:
                warnings.append('Страницы без текста пропущены: ' + ', '.join(map(str, blank)))
        pages = [p for p in pages if p['text']]
        if not pages:
            if suffix == '.pdf':
                raise ValueError('В PDF нет текстового слоя. OCR не поддерживается.')
            raise ValueError('Документ не содержит текста.')
        if sum(len(p['text']) for p in pages) > 5_000_000:
            raise ValueError('Извлечённый текст превышает 5 млн символов.')
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f'Не удалось прочитать «{name}»: {exc}') from exc
    return dict(id='file-' + digest(name.encode('utf-8') + b'\0' + raw), name=name,
                format=suffix[1:], sha256=digest(raw), pages=pages, raw=raw,
                metadata={'demo': False, 'warnings': warnings})


def corpus_fingerprint(documents):
    fields = ('id', 'name', 'format', 'sha256', 'metadata', 'pages')
    return digest(canonical([{k: d[k] for k in fields}
                             for d in sorted(documents, key=lambda d: d['id'])]).encode('utf-8'))
