"""All settings, overridable via environment variables prefixed with RAGNET_ or a .env file."""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGNET_", env_file=".env", extra="ignore")

    # ---- LLM: any OpenAI-compatible server (vLLM, Ollama `/v1`, llama.cpp, TGI) ----
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "none"
    llm_model: str = "Qwen/Qwen3-32B"
    # Cheaper/faster model for routing, query condensing, chunk contextualisation and
    # claim verification. Defaults to llm_model when unset.
    router_model: Optional[str] = None
    # Qwen3-style "thinking" is slow and interferes with tool calling; keep it off.
    llm_disable_thinking: bool = True
    llm_max_tokens: int = 2048
    llm_context_window: int = 32768
    llm_concurrency: int = 8
    llm_timeout: float = 180.0

    # ---- Embeddings / reranking (run in-process on the GPU) ----
    embed_model: str = "BAAI/bge-m3"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"  # auto | cuda | mps | cpu
    embed_batch_size: int = 64
    embed_max_length: int = 1024

    # ---- Vector store ----
    qdrant_url: Optional[str] = None  # set to use a Qdrant server; otherwise embedded local mode
    qdrant_api_key: Optional[str] = None
    collection: str = "ragnet"

    # ---- Paths ----
    data_dir: Path = Path("data")

    # ---- Chunking ----
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # ---- Retrieval ----
    prefetch_k: int = 40  # candidates from dense+sparse before reranking
    top_k: int = 6  # chunks handed to the LLM after reranking
    grep_limit: int = 12

    # ---- Ingest-time enrichment ----
    contextual: bool = False  # LLM writes a 1-2 sentence context per chunk (slower ingest, better recall)
    context_doc_tokens: int = 6000  # how much of the document the contextualiser sees
    summarize_docs: bool = True  # store an LLM summary + outline per document (used by list_documents)

    # ---- Answering ----
    verify: bool = True  # post-hoc claim-vs-citation verification
    long_context_max_tokens: int = 40000  # if the whole corpus fits, skip retrieval entirely
    history_turns: int = 6
    agent_max_steps: int = 8

    # ---- API ----
    host: str = "0.0.0.0"
    port: int = 8080

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def sessions_db(self) -> Path:
        return self.data_dir / "sessions.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def router_model_name(self) -> str:
        return self.router_model or self.llm_model


def get_settings() -> Settings:
    return Settings()
