from typing import List


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 50) -> List[str]:
    text = text.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            sentence_end = max(
                text.rfind(".", start, end),
                text.rfind("!", start, end),
                text.rfind("?", start, end),
                text.rfind("\n", start, end),
            )
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - overlap
    return chunks