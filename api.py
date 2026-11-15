from contextlib import asynccontextmanager
from fastapi import FastAPI 
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.infer import build_index, rerank_query
from src.utils import tokenize_text


MODEL_PATH = "models/reranker"
CORPUS_PATH = "data/processed/corpus.jsonl"

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description='Текст поискового запроса')
    top_k: int = Field(default=5, ge=1, le=20, description='Сколько ответов вернуть')
    bm25_k: int = Field(default=50, ge=10, le=200, description='Сколько кандидатов отобрать')

class DocumentItem(BaseModel):
    rank: int
    docid: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[DocumentItem]

@asynccontextmanager
async def lifespan(app: FastAPI):
    docs, bm25 = build_index(CORPUS_PATH)
    model = CrossEncoder(MODEL_PATH)

    app.state.docs = docs
    app.state.bm25 = bm25
    app.state.model = model

    yield

app = FastAPI(title='Neural Reranker', lifespan=lifespan)

@app.post("/search", response_model=SearchResponse)
def search(data: SearchRequest):
    docs = app.state.docs
    bm25 = app.state.bm25
    model = app.state.model

    query_tokens = tokenize_text(data.query)
    scores = bm25.get_scores(query_tokens)
    top_indicies = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:data.bm25_k]
    candidates = [docs[idx] for idx in top_indicies]

    ranked = rerank_query(model, data.query, candidates, top_k=data.top_k)

    results = [
        DocumentItem(
            rank=rank,
            docid=doc["docid"],
            score=round(float(score), 4),
            text=doc["text"],
        )
        for rank, (doc, score) in enumerate(ranked, start=1)
    ]

    return SearchResponse(query=data.query, results=results)
