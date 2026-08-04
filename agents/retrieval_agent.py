from agents.embedding_agent import EmbeddingAgent


class RetrievalAgent:
    """
    Retrieves the most relevant document chunks
    using semantic similarity search.
    """

    def __init__(self, vector_store):
        """
        vector_store: a VectorStore instance shared with the
        node that populates it in this run. This avoids loading
        a (possibly non-existent) index from disk at startup.
        """
        self.embedder = EmbeddingAgent()
        self.vector_store = vector_store

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
