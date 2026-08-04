from agents.embedding_agent import EmbeddingAgent
from utils.vectorstore import VectorStore


class RetrievalAgent:
    """
    Retrieves the most relevant document chunks
    using semantic similarity search.
    """

    def __init__(self):
        self.embedder = EmbeddingAgent()
        self.vector_store = VectorStore()

        # Load the saved FAISS index
        self.vector_store.load()

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top-k relevant chunks for a query.
        """

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results
