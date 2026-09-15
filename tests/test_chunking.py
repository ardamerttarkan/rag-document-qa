import pytest

from chunking import (
    chunk_documents,
    split_text,
)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_split_text_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_text(
            text="Örnek metin",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_split_text_creates_non_empty_chunks() -> None:
    text = " ".join(
        f"kelime{index}"
        for index in range(100)
    )

    chunks = split_text(
        text=text,
        chunk_size=80,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_split_text_returns_empty_list_for_empty_text() -> None:
    assert split_text("") == []


def test_chunk_documents_adds_expected_metadata() -> None:
    documents = [
        {
            "text": "Uyku sağlığı insan yaşamı için önemlidir.",
            "document_id": "document-1",
            "document_hash": "hash-1",
            "document_name": "uyku.pdf",
            "page": 2,
        }
    ]

    chunks = chunk_documents(
        documents=documents,
        chunk_size=800,
        overlap=150,
    )

    assert len(chunks) == 1

    metadata = chunks[0]["metadata"]

    assert metadata["document_id"] == "document-1"
    assert metadata["document_hash"] == "hash-1"
    assert metadata["document_name"] == "uyku.pdf"
    assert metadata["page"] == 2
    assert metadata["chunk_index"] == 1
    assert metadata["chunk_id"] == "document-1_2_1"
    assert metadata["section_title"] is None