import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10
QUERY_TIMEOUT_SECONDS = 120


st.set_page_config(
    page_title="RAG Doküman Soru-Cevap",
    page_icon="📄",
    layout="centered",
)

st.title("RAG Doküman Soru-Cevap Sistemi")
st.caption(
    "İndekslenmiş dokümanlara soru sorun "
    "ve kaynak temelli cevap alın."
)


def get_api_health() -> dict | None:
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        return response.json()

    except (requests.RequestException, ValueError):
        return None


health_data = get_api_health()

if health_data is None:
    st.error(
        "FastAPI servisine ulaşılamadı. "
        "Uvicorn, Qdrant ve Ollama servislerini "
        "kontrol edin."
    )
else:
    st.success("RAG API kullanıma hazır.")

    with st.expander("Sistem durumu"):
        st.write(
            "Qdrant collection:",
            health_data.get("qdrant_collection"),
        )
        st.write(
            "Embedding modeli:",
            (
                "Hazır"
                if health_data.get(
                    "embedding_model_loaded"
                )
                else "Hazır değil"
            ),
        )
        st.write(
            "Ollama:",
            (
                "Hazır"
                if health_data.get("ollama_ready")
                else "Hazır değil"
            ),
        )
        st.write(
            "LLM modeli:",
            health_data.get("llm_model"),
        )


st.divider()
st.subheader("Dokümana soru sor")

with st.form("question_form"):
    question = st.text_area(
        "Sorunuz",
        placeholder=(
            "Örnek: Düzenli uykunun insan "
            "sağlığına faydaları nelerdir?"
        ),
        height=100,
    )

    submit_button = st.form_submit_button(
        "Soruyu gönder",
        disabled=health_data is None,
    )

if submit_button:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Lütfen bir soru yazın.")
    else:
        try:
            with st.spinner("Cevap hazırlanıyor..."):
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "question": cleaned_question,
                    },
                    timeout=QUERY_TIMEOUT_SECONDS,
                )

            response.raise_for_status()
            result = response.json()

            st.subheader("Cevap")
            st.markdown(result["answer"])
            
            retrieval_latency = result.get(
                "retrieval_latency_ms"
            )
            generation_latency = result.get(
                "generation_latency_ms"
            )

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.metric(
                    "Retrieval süresi",
                    (
                        f"{retrieval_latency:.1f} ms"
                        if isinstance(
                            retrieval_latency,
                            (int, float),
                        )
                        else "-"
                    ),
                )

            with metric_columns[1]:
                st.metric(
                    "Generation süresi",
                    (
                        f"{generation_latency:.1f} ms"
                        if isinstance(
                            generation_latency,
                            (int, float),
                        )
                        else "-"
                    ),
                )

            sources = result.get("sources", [])

            st.subheader("Kaynaklar")

            if not sources:
                st.info(
                    "Bu cevap için kaynak bulunamadı."
                )
            else:
                for source in sources:
                    source_number = source.get(
                        "source_number",
                        "-",
                    )
                    document_name = source.get(
                        "document_name",
                        "Bilinmeyen doküman",
                    )
                    score = source.get("score")

                    with st.expander(
                        f"Kaynak {source_number} — "
                        f"{document_name}"
                    ):
                        st.write(
                            "Sayfa:",
                            source.get("page", "-"),
                        )
                        st.write(
                            "Chunk sırası:",
                            source.get(
                                "chunk_index",
                                "-",
                            ),
                        )
                        st.write(
                            "Chunk ID:",
                            source.get(
                                "chunk_id",
                                "-",
                            ),
                        )
                        st.write(
                            "Benzerlik skoru:",
                            (
                                f"{score:.4f}"
                                if isinstance(
                                    score,
                                    (int, float),
                                )
                                else "-"
                            ),
                        )

                        st.caption("Kaynak metni")
                        st.write(
                            source.get(
                                "text",
                                "Metin bulunamadı.",
                            )
                        )
        except requests.Timeout:
            st.error(
                "Cevap süresi aşıldı. "
                "Lütfen tekrar deneyin."
            )

        except requests.RequestException as error:
            error_message = (
                "Soru işlenirken API hatası oluştu."
            )

            if error.response is not None:
                try:
                    error_message = error.response.json().get(
                        "detail",
                        error_message,
                    )
                except ValueError:
                    pass

            st.error(error_message)

        except (ValueError, KeyError):
            st.error(
                "API'den geçerli bir cevap alınamadı."
            )