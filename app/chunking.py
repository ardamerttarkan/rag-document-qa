# Metni belirlenen boyut ve örtüşme oranına göre parçalara ayırır.
def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size sıfırdan büyük olmalıdır.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap, sıfırdan küçük veya chunk size'dan büyük olamaz."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Mümkünse kelimeyi ortadan bölmemek için
        # chunk içerisindeki son boşluğu bulur.
        if end < text_length:
            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break

        next_start = end - overlap

        # Yeni chunk'ın kelime ortasından başlamasını engeller.
        next_space = text.find(" ", next_start, end)

        if next_space != -1:
            next_start = next_space + 1

        # Döngünün aynı yerde takılmasını engeller.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

# Belgeleri metin parçalarına dönüştürür.
def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    all_chunks = []

    for document in documents:
        text_chunks = split_text(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_text in enumerate(text_chunks, start=1):
            page = document["page"]

            if page is None:
                page_label = "document"
            else:
                page_label = page

            chunk_id = (
                f"{document['document_id']}_"
                f"{page_label}_"
                f"{chunk_index}"
            )

            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "document_id": document["document_id"],
                    "document_hash": document["document_hash"],
                    "document_name": document["document_name"],
                    "page": page,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "section_title": None,
                },
            }

            all_chunks.append(chunk_data)

    return all_chunks
