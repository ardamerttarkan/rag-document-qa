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

## 3. Gün

### Chunking ve Metadata Tasarımı

Okunan dokümanların RAG sisteminde kullanılabilmesi için metinleri daha küçük parçalara ayıran chunking yapısı oluşturuldu.

- Başlangıç chunk boyutu 500 karakter olarak belirlendi.
- Chunk’lar arasında 100 karakter overlap kullanıldı.
- Kelimelerin mümkün olduğunca ortadan bölünmemesi sağlandı.
- Her doküman için `document_id` oluşturuldu.
- Dosya içeriğindeki değişiklikleri tespit etmek için SHA-256 ile `document_hash` oluşturuldu.
- Her chunk’a `chunk_id` ve sıra bilgisi eklendi.
- PDF dokümanlarında sayfa bilgisi korundu.
- TXT ve Markdown dokümanlarında sayfa bilgisi `None` olarak tutuldu.

### Chunk ve Metadata Yapısı

Oluşturulan chunk’lar metin ve metadata bölümleriyle aşağıdaki yapıda standartlaştırıldı:

```json
{
  "text": "Chunk içerisindeki doküman metni...",
  "metadata": {
    "document_id": "05e1664de46da1b1",
    "document_hash": "SHA-256 dosya özeti",
    "document_name": "uyku_duzeni.pdf",
    "page": 2,
    "chunk_id": "05e1664de46da1b1_2_2",
    "chunk_index": 2,
    "section_title": null
  }
}
```

### Duplicate ve Re-index Mantığı

Aynı dokümanın tekrar indekslenmesini önlemek amacıyla belge kimliği ve içerik hash’i üzerinden çalışan bir karar mekanizması oluşturuldu.

- Doküman daha önce indekslenmemişse `index` kararı verilir.
- Doküman kimliği ve içerik hash’i aynıysa duplicate kabul edilerek `skip` kararı verilir.
- Doküman kimliği aynı fakat içerik hash’i farklıysa `reindex` kararı verilir.
- Re-index işleminde eski dokümana ait chunk’ların silinip güncel chunk’ların yeniden eklenmesi planlandı.
- Karar mekanizması yeni, aynı ve değiştirilmiş doküman senaryolarıyla test edildi.

Gerçek ekleme, silme ve yeniden indeksleme işlemleri Qdrant entegrasyonu sırasında uygulanacaktır.

### Testler

- Chunk karakter sayılarının belirlenen sınırı aşmadığı doğrulandı.
- Gerekli metadata alanlarının bütün chunk’larda bulunduğu kontrol edildi.
- Chunk ID değerlerinin benzersiz olduğu doğrulandı.
- `index`, `skip` ve `reindex` karar senaryoları başarıyla test edildi.
