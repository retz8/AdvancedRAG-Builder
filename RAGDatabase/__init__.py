# RAGDatabase/__init__.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RAGDatabase:
    """
    LLM RAG Database (currently based on Neo4j)

    @args:
    - database_type: vector DB or knowledge graph (vector, graph)
    - embedding_model: text embedding model path (gemini api) (optional)
    """

    def __init__(self, database_type: str, embedding_model="models/text-embedding-004"):
        self.database_type = database_type
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)

        print("embedding model: ", self.embedding_model)

    def text_to_embedding(self):
        vector = self.embedding_model.embed_query("hello world")
        print("vector: ", vector[:5])
