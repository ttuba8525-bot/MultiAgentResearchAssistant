from sentence_transformers import CrossEncoder


class RerankerAgent:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, retrieved_documents, top_k=5):
        """
        Rerank retrieved chunks based on relevance.
        """

        if not retrieved_documents:
            return []

        sentence_pairs = [
            (query, doc["text"])
            for doc in retrieved_documents
        ]

        scores = self.model.predict(sentence_pairs)

        reranked = []

        for doc, score in zip(retrieved_documents, scores):

            reranked.append({
                "score": float(score),
                "text": doc["text"],
                "metadata": doc["metadata"]
            })

        reranked = sorted(
            reranked,
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked[:top_k]
