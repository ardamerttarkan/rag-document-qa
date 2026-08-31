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
""".strip()


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
            "temperature": 0.1,
        },
    )

    return response.message.content.strip()