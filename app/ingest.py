from pathlib import Path
import re

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
        return load_pdf(file_path)

    if extension in [".txt", ".md"]:
        return load_text_file(file_path)

    raise ValueError(f"Desteklenmeyen dosya türü: {extension}")


if __name__ == "__main__":
    file_path = PROJECT_ROOT / "pdf_samples" / "uyku_duzeni.pdf"

    documents = load_document(file_path)

    for document in documents:
        print("-" * 50)
        print("Doküman:", document["document_name"])
        print("Sayfa:", document["page"])
        print("Metin:")
        print(document["text"])