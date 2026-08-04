from typing import List, Dict


class CitationAgent:
    """
    Generates APA and IEEE citations from paper metadata.
    """

    @staticmethod
    def generate_ieee(papers: List[Dict]) -> List[str]:
        citations = []

        for index, paper in enumerate(papers, start=1):

            authors = paper.get("authors", "Unknown Author")
            title = paper.get("title", "Untitled")
            year = paper.get("published", "Unknown")[:4]
            url = paper.get("pdf_url") or paper.get("url", "")

            citation = (
                f'[{index}] {authors}, "{title}," '
                f'{year}. [Online]. Available: {url}'
            )

            citations.append(citation)

        return citations

    @staticmethod
    def generate_apa(papers: List[Dict]) -> List[str]:

        citations = []

        for paper in papers:

            authors = paper.get("authors", "Unknown Author")
            title = paper.get("title", "Untitled")
            year = paper.get("published", "Unknown")[:4]
            url = paper.get("pdf_url") or paper.get("url", "")

            citation = (
                f"{authors} ({year}). "
                f"{title}. Retrieved from {url}"
            )

            citations.append(citation)

        return citations

    @classmethod
    def generate(cls, papers: List[Dict], style: str = "IEEE") -> List[str]:
        """
        Dispatch to the correct citation format.
        """
        if style.upper() == "APA":
            return cls.generate_apa(papers)
        return cls.generate_ieee(papers)
