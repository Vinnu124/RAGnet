# RAGnet 2

Fast, accurate, **agentic** RAG over your own documents. Runs entirely on your own GPU machine — no API keys.
The LLM is served by any OpenAI-compatible server (vLLM, Ollama, llama.cpp); embeddings and reranking run in-process on the GPU.

```
INGEST  file ─► pymupdf4llm / docling ─► structure-aware chunks (section path, page) ─► [LLM context per chunk]
             ─► bge-m3 dense + BM25 sparse ─► Qdrant (embedded, persisted, hash-deduped)

ASK     question ─► router (condense follow-up, pick mode)
            ├─ chat        no retrieval
            ├─ longctx     whole corpus fits → feed everything, no retrieval (most accurate for small corpora)
            ├─ lookup      hybrid search → RRF → cross-encoder rerank → streamed answer with [n] citations
            └─ research    agent loop with tools: search · grep · read_page · list_documents (parallel calls)
        ─► citation verifier (each cited sentence checked against its source) ─► SQLite session memory
```

## Setup on the GPU box

```bash
# 1. deps (Python ≥3.10). torch installs the CUDA build by default on Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                      # add  --extra hifi  for Docling (OCR / hard tables),  --extra dev  for tests

# 2. serve an LLM (pick one)
#    vLLM — tool calling needs the two flags. Qwen3-32B needs ~70 GB in bf16; use Qwen/Qwen3-32B-FP8 or Qwen/Qwen3-14B on smaller cards.
uv pip install vllm
vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --gpu-memory-utilization 0.6
#    Ollama
ollama pull qwen3:32b        # then set RAGNET_LLM_BASE_URL=http://localhost:11434/v1  RAGNET_LLM_MODEL=qwen3:32b

# 3. configure
cp .env.example .env         # edit the LLM URL/model if needed

# 4. index + ask
uv run ragnet ingest docs/                       # files or folders: pdf, md, txt, html, docx
uv run ragnet ask "What was total revenue in FY2024?"
uv run ragnet chat                               # interactive, remembers the conversation
uv run ragnet serve                              # HTTP API on :8080
```

`--gpu-memory-utilization 0.6` leaves room on the same GPU for bge-m3 + the reranker (~3 GB in fp16).

## Commands

| Command | What it does |
|---|---|
| `ragnet ingest <paths…> [--hifi] [--contextual] [--force]` | Parse → chunk → embed → index. Unchanged files are skipped (SHA-256). `--hifi` uses Docling (layout model + OCR). `--contextual` has the LLM write a situating sentence per chunk before embedding (Anthropic's contextual-retrieval trick; slower ingest, noticeably better recall). |
| `ragnet docs` / `ragnet remove <id>` / `ragnet reset` | Inspect / edit the index |
| `ragnet ask "…" [--mode auto\|lookup\|research\|longctx\|chat]` | One question. `auto` lets the router decide. |
| `ragnet chat` | Interactive session with memory (`/mode`, `/clear`, `/docs`) |
| `ragnet serve` | FastAPI server with SSE streaming (`POST /ask`, `POST /ingest`, `GET /documents`, sessions) |
| `ragnet eval eval/golden.jsonl` | Retrieval hit@k, MRR, LLM-judged correctness, latency |

## How accuracy is achieved

1. **Parsing keeps structure.** Headings, tables and page numbers survive, so chunks carry `doc > section` paths and every answer cites `file · p.N`.
2. **Hybrid retrieval.** Dense (bge-m3) catches paraphrases; BM25 catches exact identifiers; Qdrant fuses them with RRF server-side.
3. **Cross-encoder reranking** (bge-reranker-v2-m3) over 40 candidates → top 6.
4. **Agentic research mode.** For multi-part / comparative questions the model plans, runs several searches in parallel, greps exact terms, reads full pages when a chunk is cut off, and only then answers.
5. **Long-context bypass.** If the whole corpus fits in the model's window, retrieval is skipped entirely — nothing to miss.
6. **Grounding rules + verifier.** Every factual sentence must cite `[n]`; a second pass checks each cited claim against its source and flags unsupported ones.
7. **Eval harness** so you can measure any change instead of guessing.

## How speed is achieved

- Index once; re-runs skip unchanged files.
- Router sends simple questions down the single-search path; only hard ones pay for the agent loop.
- Parallel tool calls inside the agent; async embedding/reranking off the event loop.
- Streaming answers on the lookup / long-context paths.
- Shared prompt prefixes (system prompt, document excerpt) so vLLM's prefix cache hits.
- `RAGNET_ROUTER_MODEL` lets a small model handle routing, chunk contexts and verification.

## API

```bash
curl -N localhost:8080/ask -H 'content-type: application/json' \
  -d '{"question":"Compare the warranty terms of model A and B","session_id":"u1"}'
# SSE events: route, status, token, answer, sources, verification, done

curl -F files=@report.pdf 'localhost:8080/ingest?contextual=true'
```

## Building a golden set

Copy `eval/golden.example.jsonl` to `eval/golden.jsonl` and write 30–50 questions from your real documents with the reference answer and the page(s) that contain it. Then `ragnet eval`. Tune `RAGNET_TOP_K`, `RAGNET_CHUNK_TOKENS`, `--contextual`, and the model choice against the numbers.

## Layout

```
ragnet/
  config.py           settings (env / .env)
  engine.py           orchestration: route → answer → verify → remember
  llm/client.py       OpenAI-compatible client (tools, streaming, robust JSON)
  ingest/             parsers.py · chunker.py · contextualizer.py · indexer.py
  retrieval/          embedder.py · store.py (Qdrant hybrid) · reranker.py · retriever.py
  agent/              prompts.py · tools.py · loop.py · router.py · evidence.py · verifier.py
  memory/sessions.py  SQLite conversation history
  api/server.py       FastAPI + SSE
  cli.py · eval.py
```

Note: embedded Qdrant (`data/qdrant`) is single-process. While `ragnet serve` is running, ingest through the API (`POST /ingest`) rather than the CLI, or point `RAGNET_QDRANT_URL` at a Qdrant server.
