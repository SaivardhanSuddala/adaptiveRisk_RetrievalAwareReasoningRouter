from chunking import chunk_text


def test_chunk_text_returns_chunks():
    chunks = chunk_text("One sentence. Two sentence. Three sentence.", chunk_size=20, overlap=5)

    assert chunks
    assert all(chunk for chunk in chunks)


def test_chunk_text_rejects_bad_overlap():
    try:
        chunk_text("hello", chunk_size=10, overlap=10)
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
