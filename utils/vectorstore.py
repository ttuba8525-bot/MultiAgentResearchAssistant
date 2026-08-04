import os
import pickle

import faiss
import numpy as np


class VectorStore:
    """
    Handles FAISS Vector Database operations.
    """

    def __init__(self, embedding_dimension=384):

        self.embedding_dimension = embedding_dimension

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        self.documents = []

    def add_documents(self, embedded_documents):
        """
        Add embedded documents to FAISS.
        """

        if not embedded_documents:
            # Nothing to add (e.g. no PDFs / no chunks this run).
            # Adding an empty array would crash FAISS.
            return

        vectors = np.array(
            [doc["embedding"] for doc in embedded_documents],
            dtype="float32"
        )

        self.index.add(vectors)

        self.documents.extend(embedded_documents)

    def similarity_search(self,
                          query_embedding,
                          top_k=5):
        """
        Perform similarity search.
        """

        if self.index.ntotal == 0:
            return []

        query = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"]
                }
            )

        return results

    def save(self,
             vector_path="data/vector_db/faiss.index",
             metadata_path="data/vector_db/documents.pkl"):
        """
        Save FAISS index and metadata.
        """

        os.makedirs("data/vector_db", exist_ok=True)

        faiss.write_index(
            self.index,
            vector_path
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self,
             vector_path="data/vector_db/faiss.index",
             metadata_path="data/vector_db/documents.pkl"):
        """
        Load FAISS index and metadata if they exist.
        Falls back to a fresh empty index instead of crashing
        when no index has been saved yet (e.g. first run).
        """

        if os.path.exists(vector_path) and os.path.exists(metadata_path):

            self.index = faiss.read_index(vector_path)

            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)

        else:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents = []
