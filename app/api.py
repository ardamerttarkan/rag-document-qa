import logging
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
from fastapi.middleware.cors import CORSMiddleware
from embeddings import load_embedding_model
from rag import answer_question
from vector_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)

logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="Dokümanlara sorulacak soru",
        examples=[
            "Düzenli uykunun insan sağlığına "
            "faydaları nelerdir?"
        ],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Soru yalnızca boşluklardan oluşamaz."
            )

        return cleaned_value


class SourceResponse(BaseModel):
    source_number: int
    document_name: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    score: float
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    retrieval_ms: float
    generation_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model, _ = load_embedding_model()
    qdrant_client = create_qdrant_client()

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: "
            f"{COLLECTION_NAME}"
        )

    app.state.embedding_model = embedding_model
    app.state.qdrant_client = qdrant_client

    yield

    qdrant_client.close()


app = FastAPI(
    title="RAG Document QA API",
    description=(
        "Dokümanlar üzerinden kaynak temelli "
        "cevap üreten RAG API"
    ),
    version="1.0.0",
    lifespan=lifespan,
)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "RAG Document QA API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
@app.get("/health")
def health_check(request: Request) -> dict:
    try:
        qdrant_client = (
            request.app.state.qdrant_client
        )

        collection_ready = (
            qdrant_client.collection_exists(
                COLLECTION_NAME
            )
        )
    except Exception as error:
        logger.exception(
            "Qdrant sağlık kontrolü başarısız."
        )

        raise HTTPException(
            status_code=503,
            detail="Qdrant servisine ulaşılamadı.",
        ) from error

    if not collection_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant collection hazır değil: "
                f"{COLLECTION_NAME}"
            ),
        )

    return {
        "status": "ok",
        "qdrant_collection": COLLECTION_NAME,
        "collection_ready": True,
        "embedding_model_loaded": True,
    }

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    payload: AskRequest,
    request: Request,
) -> dict:
    try:
        return answer_question(
            question=payload.question,
            embedding_model=(
                request.app.state.embedding_model
            ),
            qdrant_client=(
                request.app.state.qdrant_client
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "RAG sorusu işlenirken hata oluştu."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Soru işlenirken beklenmeyen "
                "bir hata oluştu."
            ),
        ) from error