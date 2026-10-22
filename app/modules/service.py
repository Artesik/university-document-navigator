"""Shared use cases for Streamlit, CLI and integration tests."""
from .config import DEFAULT_CHUNKS
from .database_manager import DatabaseManager
from .documents import corpus_fingerprint, extract_document
from .errors import IndexError
from .index_store import expected_metadata, load_index, save_index
from .search_engine import DocumentSearchEngine
from .text_processing import chunk_documents


class Navigator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.database = DatabaseManager(data_dir / 'navigator.db')
        self.database.initialize()
        self.index_path = data_dir / 'index.npz'

    def upload(self, name, raw):
        document = extract_document(name, raw)
        self.database.add_document(document)
        return document

    def expected(self, encoder, config):
        return expected_metadata(corpus_fingerprint(self.database.get_documents()), encoder.signature, config)

    def build(self, encoder, config=DEFAULT_CHUNKS, batch_size=32):
        documents = self.database.get_documents()
        fingerprint = corpus_fingerprint(documents)
        chunks = chunk_documents(documents, encoder.tokenizer, config)
        engine = DocumentSearchEngine(encoder)
        vectors = engine.build(chunks, batch_size)
        if corpus_fingerprint(self.database.get_documents()) != fingerprint:
            raise IndexError('Документы изменились во время построения. Повторите индексацию.')
        expected = expected_metadata(fingerprint, encoder.signature, config)
        metadata = save_index(self.index_path, chunks, vectors, expected, device=encoder.device,
                              precision=encoder.precision, batch_size=batch_size)
        return engine, metadata

    def open(self, encoder, config=DEFAULT_CHUNKS):
        chunks, vectors, metadata = load_index(self.index_path, self.expected(encoder, config))
        return DocumentSearchEngine(encoder, chunks, vectors), metadata
