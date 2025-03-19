from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base.config import Config

class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        self.semantic_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="interquartile")
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 2048, 
                                                                 chunk_overlap = 128,
                                                                 add_start_index = True)
    def ingest(self, path: List[Path])-> VectorStore:
        documents = []
        for docpath in path:
            loadeddocs = PyPDFium2Loader(docpath).load()
            doctext = "\n".join([doc.page_content for doc in loadeddocs])
            
            documents.extend(self.recursive_splitter.split_documents(
                self.semantic_splitter.create_documents([doctext])
            ))
        return Qdrant.from_documents(documents=documents,
                                     embedding= self.embeddings,
                                     path = Config.Path.DATABASE_DIR,
                                     url=Config.Database.QDRANT_URL,
                                     collection_name = Config.Database.DOCUMENTS_COLLECTION
                                     )
        
print("COMPLETED INGESTOR.PY")