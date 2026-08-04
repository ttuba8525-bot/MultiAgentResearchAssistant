from sentence_transformers import SentenceTransformer


class EmbeddingAgent:
    """
    Generates embeddings for document chunks using
    BAAI/bge-small-en-v1.5.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text):
        """
        Generate embedding for a single text.
        """

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_documents(self, documents):
        """
        Generate embeddings for all chunks.
        """

        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embedded_documents = []

        for doc, embedding in zip(documents, embeddings):

            embedded_documents.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "embedding": embedding
                }
            )

        return embedded_documents
