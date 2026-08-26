from pathlib import Path
import re
import pymupdf
from hashlib import sha256
from chunking import chunk_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_document_id(file_path: Path) -> str:
        relative_path = file_path.resolve().relative_to(PROJECT_ROOT)
        path_text = relative_path.as_posix()

        document_id = sha256(
        path_text.encode("utf-8")
        ).hexdigest()

        return document_id[:16]


def calculate_document_hash(file_path: Path) -> str:
    file_content = file_path.read_bytes()

    document_hash = sha256(file_content).hexdigest()

    return document_hash

#Metindeki fazla boşlukları ve boş satırları temizler
def clean_text(text: str) -> str:
    
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# PDF dosyasını sayfa sayfa okur.
    
def load_pdf(file_path: Path) -> list[dict]:
    
    pages = []

    document = pymupdf.open(file_path)

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        text = clean_text(text)

        page_data = {
            "document_name": file_path.name,
            "page": page_number,
            "text": text,
        }

        pages.append(page_data)

    document.close()

    return pages

# TXT ve MD dosyaları.
    
def load_text_file(file_path: Path) -> list[dict]:
   
    text = file_path.read_text(encoding="utf-8")
    text = clean_text(text)

    document_data = {
        "document_name": file_path.name,
        "page": None,
        "text": text,
    }

    return [document_data]

# Dosya uzantı
def load_document(file_path: Path) -> list[dict]:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        documents = load_pdf(file_path)

    elif extension in [".txt", ".md"]:
        documents = load_text_file(file_path)

    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {extension}")

    document_id = create_document_id(file_path)
    document_hash = calculate_document_hash(file_path)

    for document in documents:
        document["document_id"] = document_id
        document["document_hash"] = document_hash

    return documents




def determine_index_action(
    document: dict,
    indexed_documents: dict[str, str],
) -> str:
    document_id = document["document_id"]
    new_hash = document["document_hash"]

    old_hash = indexed_documents.get(document_id)

    if old_hash is None:
        return "index"

    if old_hash == new_hash:
        return "skip"

    return "reindex"


if __name__ == "__main__":
    file_path = PROJECT_ROOT / "pdf_samples" / "uyku_duzeni.pdf"

    documents = load_document(file_path)
    
    indexed_documents = {}

    first_document = documents[0]

    # 1. Belge henüz kayıtlı değil
    first_action = determine_index_action(
        first_document,
        indexed_documents,
    )

    print("İlk ekleme:", first_action)

    # Belgenin indekslendiğini simüle ediyoruz.
    indexed_documents[
        first_document["document_id"]
    ] = first_document["document_hash"]

    # 2. Aynı belge tekrar gönderiliyor
    second_action = determine_index_action(
        first_document,
        indexed_documents,
    )

    print("Aynı belge tekrar:", second_action)

    # 3. Aynı belgenin içeriğinin değiştiğini simüle ediyoruz.
    changed_document = first_document.copy()
    changed_document["document_hash"] = "degismis_hash"

    third_action = determine_index_action(
        changed_document,
        indexed_documents,
    )

    print("Değiştirilmiş belge:", third_action)

    chunks = chunk_documents(
        documents=documents,
        chunk_size=500,
        overlap=100,
    )

    print("Toplam chunk sayısı:", len(chunks))

    for chunk in chunks:
        metadata = chunk["metadata"]

        print("-" * 50)
        print("Chunk ID:", metadata["chunk_id"])
        print("Doküman:", metadata["document_name"])
        print("Sayfa:", metadata["page"])
        print("Chunk sırası:", metadata["chunk_index"])
        print("Karakter sayısı:", len(chunk["text"]))
        print("Metin:")
        print(chunk["text"])
        
    
        
        
    
        
        
