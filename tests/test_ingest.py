from hashlib import sha256

import pymupdf
import pytest

import ingest
from ingest import (
    clean_text,
    determine_index_action,
    load_document,
)


def test_clean_text_normalizes_whitespace() -> None:
    text = "  Düzenli   uyku\n\n\nsağlıklıdır.  "

    result = clean_text(text)

    assert result == "Düzenli uyku\n\nsağlıklıdır."


@pytest.mark.parametrize("extension", [".txt", ".md"])
def test_load_text_document(
    tmp_path,
    monkeypatch,
    extension: str,
) -> None:
    monkeypatch.setattr(
        ingest,
        "PROJECT_ROOT",
        tmp_path,
    )

    file_path = tmp_path / f"ornek{extension}"
    file_content = "Uyku düzeni sağlık için önemlidir."
    file_path.write_text(
        file_content,
        encoding="utf-8",
    )

    documents = load_document(file_path)

    assert len(documents) == 1

    document = documents[0]

    assert document["document_name"] == file_path.name
    assert document["page"] is None
    assert document["text"] == file_content
    assert len(document["document_id"]) == 16
    assert document["document_hash"] == sha256(
        file_content.encode("utf-8")
    ).hexdigest()


def test_load_pdf_page_by_page(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ingest,
        "PROJECT_ROOT",
        tmp_path,
    )

    file_path = tmp_path / "ornek.pdf"

    pdf = pymupdf.open()

    first_page = pdf.new_page()
    first_page.insert_text(
        (72, 72),
        "Birinci sayfa",
    )

    second_page = pdf.new_page()
    second_page.insert_text(
        (72, 72),
        "Ikinci sayfa",
    )

    pdf.save(file_path)
    pdf.close()

    documents = load_document(file_path)

    assert len(documents) == 2
    assert documents[0]["page"] == 1
    assert documents[1]["page"] == 2
    assert "Birinci sayfa" in documents[0]["text"]
    assert "Ikinci sayfa" in documents[1]["text"]

    assert (
        documents[0]["document_id"]
        == documents[1]["document_id"]
    )
    assert (
        documents[0]["document_hash"]
        == documents[1]["document_hash"]
    )


def test_empty_text_document_is_handled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ingest,
        "PROJECT_ROOT",
        tmp_path,
    )

    file_path = tmp_path / "bos.txt"
    file_path.write_text("", encoding="utf-8")

    documents = load_document(file_path)

    assert len(documents) == 1
    assert documents[0]["text"] == ""


def test_unsupported_extension_is_rejected(
    tmp_path,
) -> None:
    file_path = tmp_path / "ornek.docx"
    file_path.write_bytes(b"test")

    with pytest.raises(
        ValueError,
        match="Desteklenmeyen dosya türü",
    ):
        load_document(file_path)


def test_missing_text_file_is_rejected(
    tmp_path,
) -> None:
    file_path = tmp_path / "bulunamadi.txt"

    with pytest.raises(FileNotFoundError):
        load_document(file_path)


def test_corrupted_pdf_is_rejected(
    tmp_path,
) -> None:
    file_path = tmp_path / "bozuk.pdf"
    file_path.write_bytes(b"Bu gecerli bir PDF degildir.")

    with pytest.raises(pymupdf.FileDataError):
        load_document(file_path)


@pytest.mark.parametrize(
    (
        "indexed_documents",
        "new_hash",
        "expected_action",
    ),
    [
        ({}, "hash-1", "index"),
        ({"doc-1": "hash-1"}, "hash-1", "skip"),
        ({"doc-1": "hash-1"}, "hash-2", "reindex"),
    ],
)
def test_determine_index_action(
    indexed_documents: dict[str, str],
    new_hash: str,
    expected_action: str,
) -> None:
    document = {
        "document_id": "doc-1",
        "document_hash": new_hash,
    }

    result = determine_index_action(
        document=document,
        indexed_documents=indexed_documents,
    )

    assert result == expected_action