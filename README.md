# RAG Tabanlı Doküman Soru-Cevap Sistemi

## 1. Gün

python3: 3.11.2
Docker: 29.7.2, build a7dcaa6
Git: 2.39.5

## Sistem Bilgileri

- İşletim Sistemi: Debian Linux
- CPU: Intel Core i7-9750H
- CPU Çekirdek/Thread: 6 Core / 12 Thread
- RAM: 16 GB
- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GB
- NVIDIA Driver: 535.261.03
- CUDA Compatibility: 12.2
- Kullanılabilir Disk Alanı: ~57 GB

## Qdrant

Qdrant, Docker Compose ile kuruldu ve çalıştırıldı.

- Dashboard: http://localhost:6333/dashboard
- REST API Port: 6333
- gRPC Port: 6334

## 2. Gün

## Doküman Okuma ve Metin Çıkarma

PDF, TXT ve Markdown dosyalarının okunması için doküman alma (ingestion) yapısı oluşturuldu.

- PDF dosyaları PyMuPDF kullanılarak sayfa bazlı okundu.

- TXT ve Markdown (.md) dosyalarının UTF-8 formatında okunması sağlandı.

- Doküman adı ve PDF sayfa bilgisi korundu.

- Metinlerdeki gereksiz boşluklar ve satır aralıkları için temel temizleme işlemleri uygulandı.

- Dosya uzantısına göre uygun okuma fonksiyonunu seçen load_document() fonksiyonu oluşturuldu.

## Doküman Yapısı

Okunan dokümanlar aşağıdaki yapıda standartlaştırıldı:

{
"document_name": "ornek.pdf",
"page": 1,
"text": "Dokümandan çıkarılan metin..."
}
