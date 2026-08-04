import arxiv
from tavily import TavilyClient
from utils.helpers import get_tavily_api_key


class SearchAgent:
    def __init__(self):
        self.tavily = TavilyClient(api_key=get_tavily_api_key())
        self.arxiv_client = arxiv.Client()

    def search_arxiv(self, query, max_results=5):
        """
        Search research papers from arXiv.
        """

        papers = []

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for paper in self.arxiv_client.results(search):
                papers.append({
                    "title": paper.title,
                    "authors": ", ".join(
                        author.name for author in paper.authors
                    ),
                    "summary": paper.summary,
                    "published": str(paper.published.date()),
                    "pdf_url": paper.pdf_url,
                    "source": "arXiv"
                })

        except Exception as e:
            print(f"arXiv Error: {e}")

        return papers

    def search_web(self, query, max_results=5):
        """
        Search using Tavily.
        """

        try:
            response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )

            results = []

            for item in response.get("results", []):

                results.append({
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "url": item.get("url"),
                    "source": "Tavily"
                })

            return results

        except Exception as e:
            print(f"Tavily Error: {e}")
            return []

    def search(self, query):
        """
        Combined Search
        """

        arxiv_results = self.search_arxiv(query)
        tavily_results = self.search_web(query)

        return {
            "arxiv": arxiv_results,
            "web": tavily_results
        }
