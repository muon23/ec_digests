from typing import List

import pandas as pd
from datasets.search import BaseIndex
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM


class TableIndexer:
    def __init__(
            self,
            embed_model: BaseEmbedding = None,
            llm: BaseLLM = None,
            vector_store: BaseIndex = None,
    ):
        if embed_model:
            Settings.embed_model = embed_model

        if llm:
            Settings.llm = llm

        self.vector_store = vector_store if vector_store else VectorStoreIndex.from_documents([])

    def save(self, path: str):
        self.vector_store.storage_context.persist(persist_dir=path)

    @classmethod
    def load(cls, path: str) -> "TableIndexer":
        storage_context = StorageContext.from_defaults(persist_dir=path)
        return TableIndexer(vector_store=load_index_from_storage(storage_context))

    def insert(self, df: pd.DataFrame, metadata_fields: List[str] = None):

        df = df.reset_index()
        for _, row in df.iterrows():
            fields = []
            for key, value in row.dropna().to_dict().items():
                value = str(value).replace('\n', ' ')
                fields.append(f"{key}: {value}")

            text = " | ".join(fields)
            metadata = row[metadata_fields].to_dict() if metadata_fields else {}

            self.vector_store.insert(Document(text=text, metadata=metadata))

    def query(self, question: str, top_k=10, **kwarg) -> (str, dict):
        engine = self.vector_store.as_query_engine(similarity_top_k=top_k, **kwarg)
        answer = engine.query(question)
        return answer.response

    def query_detail(self, question: str, top_k=5, **kwarg) -> (str, dict):
        engine = self.vector_store.as_query_engine(similarity_top_k=top_k, **kwarg)
        answer = engine.query(question)
        return answer





