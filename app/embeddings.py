from math import isfinite
from time import perf_counter

from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from ingest import PROJECT_ROOT, load_document


MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 8


# Embedding modelini yükler ve yükleme süresini ölçer.
def load_embedding_model() -> tuple[SentenceTransformer, float]:
    start_time = perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    load_duration = perf_counter() - start_time

    return model, load_duration


# Metin parçaları için embedding vektörleri üretir.
def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict], float]:
    if not chunks:
        raise ValueError("Embedding üretilecek chunk bulunamadı.")

    if batch_size <= 0:
        raise ValueError("Batch size sıfırdan büyük olmalıdır.")

    passages = []

    for chunk in chunks:
        text = chunk["text"].strip()

        if not text:
            raise ValueError(
                f"Boş chunk bulundu: {chunk['metadata']['chunk_id']}"
            )

        passages.append(f"passage: {text}")

    start_time = perf_counter()

    vectors = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embedding_duration = perf_counter() - start_time

    expected_shape = (
        len(chunks),
        EMBEDDING_DIMENSION,
    )

    if vectors.shape != expected_shape:
        raise ValueError(
            f"Beklenen shape {expected_shape}, "
            f"gelen shape {vectors.shape}"
        )

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        vector_list = vector.tolist()

        if not all(isfinite(value) for value in vector_list):
            raise ValueError(
                "Embedding içerisinde geçersiz sayı bulundu."
            )

        embedded_chunk = {
            "text": chunk["text"],
            "embedding": vector_list,
            "metadata": chunk["metadata"].copy(),
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks, embedding_duration


if __name__ == "__main__":
    file_path = (
        PROJECT_ROOT
        / "pdf_samples"
        / "uyku_duzeni.pdf"
    )

    documents = load_document(file_path)

    chunks = chunk_documents(
        documents=documents,
        chunk_size=800,
        overlap=150,
    )

    print("Toplam chunk sayısı:", len(chunks))

    model, model_load_duration = load_embedding_model()

    embedded_chunks, embedding_duration = embed_chunks(
        chunks=chunks,
        model=model,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    average_duration = (
        embedding_duration / len(embedded_chunks)
    )

    chunks_per_second = (
        len(embedded_chunks) / embedding_duration
    )

    first_chunk = embedded_chunks[0]

    print("Model:", MODEL_NAME)
    print("Cihaz:", model.device)
    print("Batch size:", DEFAULT_BATCH_SIZE)
    print(
        f"Model yükleme süresi: "
        f"{model_load_duration:.3f} saniye"
    )
    print(
        f"Toplam embedding süresi: "
        f"{embedding_duration:.3f} saniye"
    )
    print(
        f"Chunk başına ortalama süre: "
        f"{average_duration * 1000:.3f} ms"
    )
    print(
        f"Saniyede işlenen chunk: "
        f"{chunks_per_second:.2f}"
    )
    print(
        "Embedding shape:",
        (
            len(embedded_chunks),
            len(first_chunk["embedding"]),
        ),
    )
    print(
        "İlk chunk ID:",
        first_chunk["metadata"]["chunk_id"],
    )
    print(
        "İlk vektörün ilk 5 değeri:",
        first_chunk["embedding"][:5],
    )

    print("Embedding kontrolleri başarılı.")
