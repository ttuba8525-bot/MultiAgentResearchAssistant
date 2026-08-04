from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkAgent:
    """
    Splits extracted text into smaller chunks for embeddings.
    """

    def __init__(self,
                 chunk_size=1000,
                 chunk_overlap=200):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk_text(self, text):
        """
        Split a complete document into chunks.
        """

        return self.text_splitter.split_text(text)

    def chunk_sections(self, sections):
        """
        Split each academic section separately while
        preserving metadata.
        """

        documents = []

        for section_name, content in sections.items():

            if not content.strip():
                continue

            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):

                documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "section": section_name,
                            "chunk_id": i
                        }
                    }
                )

        return documents

    def process(self, parsed_pdf):
        """
        Complete chunking pipeline.
        """

        return self.chunk_sections(parsed_pdf["sections"])
