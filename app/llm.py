import ollama



MODEL_NAME = "qwen3:4b-instruct"

SYSTEM_PROMPT = """
Sen bir doküman soru-cevap asistanısın.

Kurallar:
1. Yalnızca kullanıcı tarafından verilen bağlamdaki bilgileri kullan.
2. Soruyu doğrudan cevapla; sorulmayan ek bilgileri cevaba ekleme.
3. Bağlamda cevap yoksa yalnızca "Bu sorunun cevabı verilen dokümanlarda bulunamadı." de.
4. Cevaptaki her madde veya bilgi sonuna dayandığı kaynak numarasını ekle.
5. Kaynakları yalnızca [Kaynak 1], [Kaynak 2] gibi bağlamda verilen biçimde yaz.
6. "[Bağlam 1]" gibi farklı veya uydurma kaynak ifadeleri kullanma.
7. Bağlamda bulunmayan kaynak numaralarını kullanma.
8. Türkçe, açık ve kısa cevap ver.
9. Sorunun istediği bilgi türüne uymayan içerikleri ekleme. Örneğin faydalar soruluyorsa olumsuz sonuçları listeleme.
10. Bağlamdaki kesinlik düzeyini koru. "Katkıda bulunur", "yardımcı olur" veya "etkileyebilir" ifadelerini "önler", "kesinlikle sağlar" ya da "neden olur" gibi daha kesin ifadelere dönüştürme.
11. Tamamı büyük harflerden oluşan cümleler kullanma.
12. Cevabı yazım ve dil bilgisi kurallarına uygun oluştur.
13. Cevabı en fazla dört kısa madde veya dört kısa cümleyle sınırlandır.
14. Kullanıcının sorduğu bilgi türünün tersindeki bilgileri ekleme. Örneğin faydalar soruluyorsa uykusuzluğun zararlarını cevaba dahil etme.
""".strip()



def is_model_available(
    model_name: str = MODEL_NAME,
) -> bool:
    response = ollama.list()

    installed_models = {
        model.model
        for model in response.models
    }

    return model_name in installed_models

# Verilen soru ve bağlama göre model cevabı üretir.
def generate_answer(
    question: str,
    context: str,
) -> str:
    cleaned_question = question.strip()
    cleaned_context = context.strip()

    if not cleaned_question:
        raise ValueError("Kullanıcı sorusu boş olamaz.")

    if not cleaned_context:
        return (
            "Bu sorunun cevabı verilen "
            "dokümanlarda bulunamadı."
        )

    user_prompt = f"""
BAĞLAM BAŞLANGICI

{cleaned_context}

BAĞLAM SONU

SORU:
{cleaned_question}
""".strip()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        stream=False,
        options={
            "temperature": 0.0,
        },
    )

    return response.message.content.strip()
