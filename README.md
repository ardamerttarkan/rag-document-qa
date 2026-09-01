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

## 4. Gün

### Embedding Modelinin Kurulması

Doküman parçalarının anlamsal olarak temsil edilebilmesi için `sentence-transformers` kütüphanesi ve `intfloat/multilingual-e5-small` embedding modeli kullanıldı.

- Model CPU üzerinde çalıştırıldı.
- Modelin ürettiği vektör boyutu 384 olarak doğrulandı.
- Doküman parçalarının başına E5 modelinin beklediği `passage:` ön eki eklendi.
- Chunk’lar toplu olarak işlenebilmesi için batch yapısı kullanıldı.
- Vektörler cosine similarity ile kullanılmaya uygun olacak şekilde normalize edildi.

### Embedding İşlemi

Önceki gün oluşturulan doküman okuma ve chunking yapısı embedding sistemiyle birleştirildi.

İşlem sırası:

1. Doküman okundu ve temizlendi.
2. Doküman belirlenen boyutlara göre chunk’lara ayrıldı.
3. Her chunk’ın metni embedding modeline gönderildi.
4. Her chunk için 384 boyutlu bir vektör üretildi.
5. Üretilen vektörler chunk metadata bilgileriyle eşleştirildi.

Her embedding kaydında aşağıdaki bilgiler korundu:

- Doküman kimliği
- Doküman hash değeri
- Doküman adı
- Sayfa numarası
- Chunk kimliği
- Chunk sırası
- Bölüm başlığı
- Chunk metni
- 384 boyutlu embedding vektörü

### Kontroller

Embedding işlemi sonrasında aşağıdaki kontroller uygulandı:

- Üretilen embedding sayısının chunk sayısıyla aynı olduğu kontrol edildi.
- Her embedding vektörünün 384 boyutunda olduğu doğrulandı.
- Vektörlerin geçerli ve sonlu sayılardan oluştuğu kontrol edildi.
- Chunk metadata bilgilerinin embedding işleminden sonra korunduğu doğrulandı.

### Performans Ölçümü

pdf dosyası üzerinde yapılan test sonucunda:

- Toplam chunk sayısı: 11
- Batch size: 8
- Kullanılan cihaz: CPU
- Embedding shape: `(11, 384)`
- Toplam embedding süresi: `0.342 saniye`
- Chunk başına ortalama süre: `31.113 ms`
- Saniyede işlenen chunk sayısı: `32.14`

İlk çalıştırmada modelin indirilmesi ve yüklenmesi nedeniyle model yükleme süresi yaklaşık `59.637 saniye` olarak ölçüldü. Model dosyaları yerel önbelleğe kaydedildiği için sonraki çalıştırmalarda tekrar indirme yapılmasına gerek kalmadı.

### Gün Sonu Çıktısı

Dokümanlardan oluşturulan chunk’lar, 384 boyutlu anlamsal vektörlere dönüştürüldü. Böylece chunk’ların Qdrant vektör veritabanına kaydedilebilmesi ve anlamsal benzerlik aramasında kullanılabilmesi için gerekli embedding aşaması tamamlandı.

## 5. Gün

### Qdrant ile Vektör Kayıt ve Semantic Search

Doküman chunk’larından üretilen embedding vektörlerinin saklanması için Qdrant bağlantısı oluşturuldu.

- `rag_documents` adında bir collection oluşturuldu.
- Collection, 384 boyutlu vektörler ve Cosine benzerlik yöntemiyle yapılandırıldı.
- Her chunk; UUID, embedding vektörü ve payload bilgileriyle Qdrant’a kaydedildi.
- Payload içerisinde chunk metni, doküman adı, sayfa numarası ve diğer metadata bilgileri korundu.
- Upsert işlemi kullanılarak aynı dokümanın tekrar eklenmesi durumunda duplicate point oluşması engellendi.
- İlk ve ikinci çalıştırmada collection içerisindeki point sayısının 11 kaldığı doğrulandı.

### Retrieval İşlemi

Kullanıcı sorularını embedding’e dönüştüren ve Qdrant üzerinde anlamsal arama yapan retrieval yapısı oluşturuldu.

- Doküman metinlerinde `passage:` ön eki kullanıldı.
- Kullanıcı sorularında `query:` ön eki kullanıldı.
- Kullanıcıdan dinamik olarak soru alınması sağlandı.
- En ilgili üç chunk’ın getirilmesi için Top-K değeri `3` olarak belirlendi.
- Sonuçlarda benzerlik skoru, doküman adı, sayfa, chunk kimliği ve metin gösterildi.
- Alakasız sonuçların filtrelenmesi için başlangıç skor eşiği `0.80` olarak ayarlandı.

İlgili bir uyku sorusunda en yüksek benzerlik skoru `0.9070` olarak ölçüldü. Alakasız İstanbul sorusunda skor `0.7625` seviyesinde kaldı ve skor eşiği sayesinde sonuç gösterilmedi.

### Gün Sonu Çıktısı

LLM kullanılmadan çalışan retrieval sistemi tamamlandı. Kullanıcı soruları embedding vektörüne dönüştürülerek Qdrant içerisinde anlamsal olarak en yakın doküman parçalarının bulunması sağlandı.

## 6. Gün – Yerel LLM Entegrasyonu

Ollama üzerinden Qwen3 4B Instruct modeli yerel olarak çalıştırıldı. Qdrant’tan getirilen ilgili metin parçaları context hâline getirilerek modele gönderildi ve doküman içeriğine dayalı cevap üretimi sağlandı.

Modelin kullandığı kaynakları belirtmesi ve dokümanda bulunmayan bilgiler için cevap üretmemesi amacıyla uygun prompt kuralları oluşturuldu. Alakasız sorular benzerlik eşiğiyle filtrelendi.

Sistem `python3 app/rag.py` komutuyla çalıştırılabilir.

## 7. Gün – RAG Değerlendirme ve Optimizasyon

RAG sisteminin retrieval ve generation aşamalarını ölçmek amacıyla 12 sorudan oluşan bir değerlendirme veri seti hazırlandı. Veri setinde 6 cevaplanabilir, 3 konuyla ilgili fakat cevapsız ve 3 tamamen alakasız soru kullanıldı.

Retrieval değerlendirmesinde `Top-K` için 1, 3 ve 5; benzerlik eşiği için 0.75, 0.80 ve 0.85 değerleri karşılaştırıldı. `Top-K=1` kullanıldığında iki doğru içerik kaçırılırken `Top-K=3` ve `Top-K=5` aynı başarıyı verdi. Daha az context ve daha düşük gürültü nedeniyle nihai ayarlar `Top-K=3` ve `score_threshold=0.80` olarak belirlendi.

Generation değerlendirmesinde cevapların beklenen anahtar kelimeleri ve geçerli kaynak numaralarını içerip içermediği kontrol edildi. Dokümanda cevabı bulunmayan sorularda modelin güvenli ret mesajı vermesi, tamamen alakasız sorularda ise LLM’in hiç çağrılmaması doğrulandı. Otomatik testlerde 12 sorunun tamamı başarılı oldu ve yalnızca 8 soru için LLM çağrısı yapıldı.

Manuel incelemede otomatik kontrollerin yakalayamadığı bazı dil ve anlam sapmaları görülmesi üzerine system prompt geliştirildi. Cevap uzunluğu dört maddeyle sınırlandırıldı, kaynaktaki kesinlik düzeyinin korunması sağlandı ve soruyla ilgisiz bilgilerin cevaba eklenmemesi için yeni kurallar tanımlandı.

Bu aşamada aşağıdaki değerlendirme dosyaları oluşturuldu:

- `evaluation/questions.json`
- `app/evaluation.py`
- `app/compare_retrieval.py`
- `app/evaluate_generation.py`
- `evaluation/generation_results.json`

Son testte retrieval başarı oranı %100, otomatik generation başarı oranı %100, ortalama retrieval süresi 42.7 ms ve ortalama generation süresi 4.72 saniye olarak ölçüldü.
