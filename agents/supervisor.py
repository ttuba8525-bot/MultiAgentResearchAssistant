from typing import TypedDict, List, Dict, Optional


class ResearchState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    """

    # User Input
    query: str
    uploaded_files: List[str]

    # Search Results
    search_results: List[Dict]

    # Parsed Documents
    parsed_documents: List[Dict]

    # Chunks
    chunks: List[Dict]

    # Embedded Documents
    embedded_documents: List[Dict]

    # Retrieved Context
    retrieved_documents: List[Dict]

    # Reranked Context
    reranked_documents: List[Dict]

    # Outputs
    literature_review: Optional[str]
    research_plan: Optional[str]
    research_paper: Optional[str]

    # Citations
    citations: List[str]


class SupervisorAgent:
    """
    Creates the initial workflow state.
    """

    @staticmethod
    def initialize_state(query: str, uploaded_files: List[str]) -> ResearchState:

        return {
            "query": query,
            "uploaded_files": uploaded_files,

            "search_results": [],

            "parsed_documents": [],

            "chunks": [],

            "embedded_documents": [],

            "retrieved_documents": [],

            "reranked_documents": [],

            "literature_review": None,

            "research_plan": None,

            "research_paper": None,

            "citations": []
        }
