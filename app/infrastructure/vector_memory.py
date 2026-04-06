import uuid

import chromadb
from sentence_transformers import SentenceTransformer


class VectorMemory:
    def __init__(self, logger):
        self.logger = logger
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="zentris_memory")
        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def save_memory(self, user_id: str, text: str) -> None:
        try:
            embedding = self.embedding_model.encode(text).tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"user_id": str(user_id)}],
                ids=[f"{user_id}-{uuid.uuid4().hex}"],
            )
        except Exception as exc:
            self.logger.warning("No se pudo guardar memoria vectorial: %s", exc)

    def get_relevant_memories(self, user_id: str, query: str) -> str:
        try:
            embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=2,
                where={"user_id": str(user_id)},
            )
            docs = results.get("documents", [[]])[0]
            return "\n".join(docs) if docs else ""
        except Exception as exc:
            self.logger.warning("No se pudo consultar memoria vectorial: %s", exc)
            return ""
