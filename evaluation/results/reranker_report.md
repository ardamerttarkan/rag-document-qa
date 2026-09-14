# Reranker Benchmark ve Hata Analizi

## Yapılandırma

- Chunk size: `500`
- Overlap: `100`
- Top-K: `5`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Çalışma cihazı: `CPU`

## Benchmark Sonuçları

| Yöntem | Recall@1 | Recall@3 | Recall@5 | MRR | Ortalama retrieval |
|---|---:|---:|---:|---:|---:|
| Dense | 68.89% | 86.67% | 88.89% | 0.7759 | 2.91 ms |
| Hybrid | 73.33% | 84.44% | 86.67% | 0.7833 | 3.21 ms |
| Hybrid + Reranker | 80.00% | 84.44% | 86.67% | 0.8278 | 130.25 ms |

Ortalama yalnızca reranker inference süresi: 127.05 ms.

## Hata Analizi

- İyileşen soru: `4`
- Gerileyen soru: `2`
- Değişmeyen soru: `39`

| Soru | Durum | Hybrid sıra | Reranker sıra |
|---|---|---:|---:|
| Q009 | improved | 3 | 1 |
| Q013 | improved | 2 | 1 |
| Q014 | improved | 4 | 1 |
| Q019 | degraded | 3 | 4 |
| Q025 | improved | 3 | 1 |
| Q030 | degraded | 1 | 2 |

## İlk Yorum

Reranker yalnızca hybrid retrieval tarafından getirilen ilk 5 sonucu yeniden sıralamıştır. Bu nedenle aday kümesinde bulunmayan doğru bir chunk'ı geri getiremez; esas etkisi Recall@1, Recall@3 ve MRR üzerinde görülür. Nihai kullanım kararı başarı artışı ile CPU gecikmesi birlikte değerlendirilerek verilmelidir.
