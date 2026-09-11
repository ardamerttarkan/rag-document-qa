# Dense ve Hybrid Retrieval Karşılaştırması

## Yapılandırma

- Chunk size: `500`
- Overlap: `100`
- Top-K: `5`
- Hybrid yöntem: Dense + BM25 + Reciprocal Rank Fusion

## Sonuçlar

| Yöntem | Recall@1 | Recall@3 | Recall@5 | MRR | Ortalama retrieval |
|---|---:|---:|---:|---:|---:|
| Dense | 68.89% | 86.67% | 88.89% | 0.7759 | 5.04 ms |
| Hybrid | 73.33% | 84.44% | 86.67% | 0.7833 | 5.58 ms |

## İlk Yorum

Hybrid yöntemin Recall@3 değişimi: -2.22%.

Hybrid yöntemin MRR değişimi: +0.0074.

Hybrid retrieval, anlamsal dense arama ile kelime tabanlı BM25 sonuçlarını RRF yöntemiyle birleştirmiştir. Nihai yöntem seçilirken retrieval başarısı ile gecikme birlikte değerlendirilmelidir.
