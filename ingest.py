from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import get_settings

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}


def load_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == '.pdf':
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding='utf-8').load()
        for doc in loaded:
            doc.metadata['source'] = path.name
        documents.extend(loaded)
    return documents


def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = (text or '').strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def prepare_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks: list[Document] = []
    for doc in documents:
        for part in split_text_recursive(doc.page_content, chunk_size, chunk_overlap):
            chunks.append(Document(page_content=part, metadata=dict(doc.metadata)))
    for idx, doc in enumerate(chunks):
        doc.metadata['chunk_id'] = idx
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.strip()]


def ingest() -> None:
    settings = get_settings()
    data_dir = settings.data_path
    index_dir = settings.index_path
    faiss_dir = index_dir / 'faiss_index'
    chunks_path = index_dir / 'chunks.pkl'
    bm25_tokens_path = index_dir / 'bm25_tokens.pkl'

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading documents from: {data_dir}')
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f'No supported documents found in {data_dir}')
    print(f'Loaded {len(documents)} document pages/files')

    chunks = prepare_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    print(f'Prepared {len(chunks)} chunks')

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    print(f'Building FAISS index with embeddings model: {settings.embedding_model}')
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(faiss_dir))

    with chunks_path.open('wb') as f:
        pickle.dump(chunks, f)

    tokenized_corpus = [tokenize_for_bm25(doc.page_content) for doc in chunks]
    with bm25_tokens_path.open('wb') as f:
        pickle.dump(tokenized_corpus, f)

    print(f'Saved vector index to: {faiss_dir}')
    print(f'Saved chunks to: {chunks_path}')
    print(f'Saved BM25 tokens to: {bm25_tokens_path}')


if __name__ == '__main__':
    ingest()
