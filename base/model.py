from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
import warnings
warnings.filterwarnings("ignore")

from base.config import Config
from flashrank import Ranker

ranker = Ranker(model_name=Config.Model.RERANKER)


def create_llm() -> BaseLanguageModel:
        return ChatOllama(
            model=Config.Model.LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )


def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER, client=ranker)


print("COMPLETED MODEL.PY")