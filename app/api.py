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
    list_indexed_documents,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from llm import (
    MODEL_NAME,
    is_model_available,
)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_handler = logging.StreamHandler()

    log_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

logger.propagate = False

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
    retrieval_latency_ms: float
    generation_latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
    
class DocumentResponse(BaseModel):
    document_id: str
    document_name: str | None = None
    document_hash: str | None = None
    chunk_count: int


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentResponse]


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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    error: RequestValidationError,
):
    error_summary = [
        {
            "location": ".".join(
                str(part)
                for part in item["loc"]
            ),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]

    logger.warning(
        "İstek doğrulama hatası | "
        "method=%s | path=%s | errors=%s",
        request.method,
        request.url.path,
        error_summary,
    )

    return await request_validation_exception_handler(
        request,
        error,
    )
    
@app.get("/")
def root() -> dict:
    return {
        "message": "RAG Document QA API",
        "documentation": "/docs",
        "health": "/health",
    }



@app.get(
    "/health",
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Bağımlı servis hazır değil",
        },
    },
)
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

    try:
        ollama_ready = is_model_available(
            MODEL_NAME
        )
    except Exception as error:
        logger.exception(
            "Ollama sağlık kontrolü başarısız."
        )

        raise HTTPException(
            status_code=503,
            detail="Ollama servisine ulaşılamadı.",
        ) from error

    if not ollama_ready:
        logger.warning(
            "Ollama modeli bulunamadı | model=%s",
            MODEL_NAME,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama modeli hazır değil: "
                f"{MODEL_NAME}"
            ),
        )

    logger.info(
        "Sağlık kontrolü başarılı | "
        "collection=%s | model=%s",
        COLLECTION_NAME,
        MODEL_NAME,
    )

    return {
        "status": "ok",
        "qdrant_collection": COLLECTION_NAME,
        "collection_ready": True,
        "embedding_model_loaded": True,
        "ollama_ready": True,
        "llm_model": MODEL_NAME,
    }

@app.get(
    "/documents",
    response_model=DocumentListResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": (
                "Dokümanlar listelenemedi"
            ),
        },
    },
)
def get_documents(request: Request) -> dict:
    try:
        documents = list_indexed_documents(
            request.app.state.qdrant_client
        )

        total_chunks = sum(
            document["chunk_count"]
            for document in documents
        )

        logger.info(
            "Dokümanlar listelendi | "
            "document_count=%d | chunk_count=%d",
            len(documents),
            total_chunks,
        )

        return {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "documents": documents,
        }

    except Exception as error:
        logger.exception(
            "Dokümanlar listelenirken hata oluştu."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Dokümanlar listelenirken "
                "beklenmeyen bir hata oluştu."
            ),
        ) from error

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
    logger.info(
        "Sağlık kontrolü başarılı | "
        "collection=%s",
        COLLECTION_NAME,
    )
    return {
        "status": "ok",
        "qdrant_collection": COLLECTION_NAME,
        "collection_ready": True,
        "embedding_model_loaded": True,
    }

@app.post(
    "/query",
    response_model=AskResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Geçersiz soru",
        },
        500: {
            "model": ErrorResponse,
            "description": "RAG işlemi sırasında sunucu hatası",
        },
    },
)
@app.post(
    "/ask",
    response_model=AskResponse,
    include_in_schema=False,
)
def ask_question(
    payload: AskRequest,
    request: Request,
) -> dict:
    try:
        result = answer_question(
            question=payload.question,
            embedding_model=(
                request.app.state.embedding_model
            ),
            qdrant_client=(
                request.app.state.qdrant_client
            ),
        )

        retrieval_latency_ms = result["retrieval_ms"]

        generation_latency_ms = round(
            result["generation_seconds"] * 1000,
            2,
        )

        source_count = len(result["sources"])

        if source_count == 0:
            logger.warning(
                "Soru için kaynak bulunamadı | "
                "question=%r | "
                "retrieval_latency_ms=%.2f",
                payload.question,
                retrieval_latency_ms,
            )
        else:
            logger.info(
                "Soru başarıyla işlendi | "
                "question=%r | "
                "source_count=%d | "
                "retrieval_latency_ms=%.2f | "
                "generation_latency_ms=%.2f",
                payload.question,
                source_count,
                retrieval_latency_ms,
                generation_latency_ms,
            )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieval_latency_ms": (
                retrieval_latency_ms
            ),
            "generation_latency_ms": (
                generation_latency_ms
            ),
        }

    except ValueError as error:
        logger.warning(
            "Geçersiz soru | question=%r | error=%s",
            payload.question,
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "RAG sorusu işlenirken hata oluştu | "
            "question=%r",
            payload.question,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Soru işlenirken beklenmeyen "
                "bir hata oluştu."
            ),
        ) from error
