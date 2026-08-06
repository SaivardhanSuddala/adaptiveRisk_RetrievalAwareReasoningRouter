from pathlib import Path

from chunking import chunk_text
from embeddings import embed_documents
from vector_store import add_documents


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def load_and_index_documents(directory: str) -> int:

    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(directory)

    indexed = 0

    for file in root.rglob("*"):

        if (
            not file.is_file()
            or file.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue

        text = file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if not text:
            continue

        chunks = chunk_text(text)

        embeddings = embed_documents(chunks)

        ids = [
            f"{file.stem}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": str(file),
                "chunk": i,
            }
            for i in range(len(chunks))
        ]

        add_documents(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        indexed += len(chunks)

    return indexed