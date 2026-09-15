from types import SimpleNamespace

import pytest

from embeddings import EMBEDDING_DIMENSION
from retrieval import (
    embed_query,
    search_similar_chunks,
)
from vector_store import COLLECTION_NAME


class FakeVector:
    def __init__(self, values: list[float]):
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeModel:
    def __init__(self, values: list[float]):
        self.values = values
        self.received_text = None
        self.received_options = None

    def encode(self, text: str, **options):
        self.received_text = text
        self.received_options = options

        return FakeVector(self.values)


class FakeQdrantClient:
    def __init__(self, points: list):
        self.points = points
        self.received_options = None

    def query_points(self, **options):
        self.received_options = options

        return SimpleNamespace(points=self.points)


def test_embed_query_uses_expected_prefix_and_options() -> None:
    model = FakeModel(
        [0.0] * EMBEDDING_DIMENSION
    )

    vector = embed_query(
        query="  Uyku neden önemlidir?  ",
        model=model,
    )

    assert len(vector) == EMBEDDING_DIMENSION
    assert model.received_text == (
        "query: Uyku neden önemlidir?"
    )
    assert model.received_options == {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


@pytest.mark.parametrize(
    "query",
    ["", "   ", "\n\t"],
)
def test_embed_query_rejects_blank_query(
    query: str,
) -> None:
    model = FakeModel(
        [0.0] * EMBEDDING_DIMENSION
    )

    with pytest.raises(
        ValueError,
        match="Kullanıcı sorusu boş olamaz",
    ):
        embed_query(
            query=query,
            model=model,
        )


def test_embed_query_rejects_wrong_dimension() -> None:
    model = FakeModel([0.0] * 10)

    with pytest.raises(
        ValueError,
        match="Beklenen query vektör boyutu",
    ):
        embed_query(
            query="Örnek soru",
            model=model,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf")],
)
def test_embed_query_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    values = [0.0] * EMBEDDING_DIMENSION
    values[0] = invalid_value

    model = FakeModel(values)

    with pytest.raises(
        ValueError,
        match="geçersiz sayı",
    ):
        embed_query(
            query="Örnek soru",
            model=model,
        )


def test_search_similar_chunks_calls_qdrant() -> None:
    expected_points = [
        SimpleNamespace(
            id="point-1",
            score=0.91,
        )
    ]

    client = FakeQdrantClient(expected_points)
    query_vector = [0.0] * EMBEDDING_DIMENSION

    results = search_similar_chunks(
        client=client,
        query_vector=query_vector,
        top_k=5,
        score_threshold=0.75,
    )

    assert results == expected_points
    assert client.received_options == {
        "collection_name": COLLECTION_NAME,
        "query": query_vector,
        "limit": 5,
        "score_threshold": 0.75,
        "with_payload": True,
        "with_vectors": False,
    }


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(
    top_k: int,
) -> None:
    client = FakeQdrantClient([])

    with pytest.raises(
        ValueError,
        match="Top-K sıfırdan büyük olmalıdır",
    ):
        search_similar_chunks(
            client=client,
            query_vector=(
                [0.0] * EMBEDDING_DIMENSION
            ),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "score_threshold",
    [-1.01, 1.01],
)
def test_search_rejects_invalid_threshold(
    score_threshold: float,
) -> None:
    client = FakeQdrantClient([])

    with pytest.raises(
        ValueError,
        match="Benzerlik eşiği",
    ):
        search_similar_chunks(
            client=client,
            query_vector=(
                [0.0] * EMBEDDING_DIMENSION
            ),
            score_threshold=score_threshold,
        )


def test_search_rejects_wrong_vector_dimension() -> None:
    client = FakeQdrantClient([])

    with pytest.raises(
        ValueError,
        match="384 boyutunda olmalıdır",
    ):
        search_similar_chunks(
            client=client,
            query_vector=[0.0] * 10,
        )