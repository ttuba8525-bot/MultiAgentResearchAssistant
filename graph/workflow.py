from langgraph.graph import StateGraph, START, END

from agents.supervisor import ResearchState

from agents.search_agent import SearchAgent
from agents.parser_agent import ParserAgent
from agents.chunk_agent import ChunkAgent
from agents.embedding_agent import EmbeddingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reranker_agent import RerankerAgent
from agents.reviewer_agent import ReviewerAgent
from agents.planner_agent import PlannerAgent
from agents.writer_agent import WriterAgent
from agents.citation_agent import CitationAgent

from utils.vectorstore import VectorStore


# Initialize Agents
search_agent = SearchAgent()
parser_agent = ParserAgent()
chunk_agent = ChunkAgent()
embedding_agent = EmbeddingAgent()
reranker_agent = RerankerAgent()
reviewer_agent = ReviewerAgent()
planner_agent = PlannerAgent()
writer_agent = WriterAgent()
citation_agent = CitationAgent()

# One VectorStore instance shared between the node that fills it
# (vectorstore_node) and the RetrievalAgent that reads from it.
# This avoids RetrievalAgent trying to load a not-yet-created
# index from disk on the very first run.
vector_store = VectorStore()
retrieval_agent = RetrievalAgent(vector_store)


# -----------------------------
# Nodes
# -----------------------------

def search_node(state: ResearchState):

    results = search_agent.search(state["query"])

    state["search_results"] = results["arxiv"]

    return state


def parser_node(state: ResearchState):

    parsed_documents = []

    # Parse any uploaded PDFs
    for pdf in state["uploaded_files"]:

        parsed = parser_agent.parse_pdf(pdf)

        parsed_documents.append(parsed)

    # Also turn arXiv abstracts into "documents" so the pipeline
    # still has something to chunk/embed/retrieve/review even when
    # the user hasn't uploaded any PDFs of their own.
    for paper in state["search_results"]:

        summary = paper.get("summary", "")

        if not summary.strip():
            continue

        parsed_documents.append(
            {
                "full_text": summary,
                "sections": {
                    "Abstract": f'{paper.get("title", "")}\n{summary}'
                }
            }
        )

    state["parsed_documents"] = parsed_documents

    return state


def chunk_node(state: ResearchState):

    chunks = []

    for document in state["parsed_documents"]:

        chunks.extend(
            chunk_agent.process(document)
        )

    state["chunks"] = chunks

    return state


def embedding_node(state: ResearchState):

    if not state["chunks"]:
        state["embedded_documents"] = []
        return state

    embedded = embedding_agent.embed_documents(
        state["chunks"]
    )

    state["embedded_documents"] = embedded

    return state


def vectorstore_node(state: ResearchState):

    vector_store.add_documents(
        state["embedded_documents"]
    )

    vector_store.save()

    return state


def retrieval_node(state: ResearchState):

    retrieved = retrieval_agent.retrieve(
        state["query"],
        top_k=10
    )

    state["retrieved_documents"] = retrieved

    return state


def reranker_node(state: ResearchState):

    reranked = reranker_agent.rerank(
        state["query"],
        state["retrieved_documents"],
        top_k=5
    )

    state["reranked_documents"] = reranked

    return state


def reviewer_node(state: ResearchState):

    review = reviewer_agent.review(
        state["query"],
        state["reranked_documents"]
    )

    state["literature_review"] = review

    return state


def planner_node(state: ResearchState):

    plan = planner_agent.generate_plan(
        state["query"],
        state["literature_review"]
    )

    state["research_plan"] = plan

    return state


def writer_node(state: ResearchState):

    paper = writer_agent.write_paper(
        state["query"],
        state["literature_review"],
        state["research_plan"]
    )

    state["research_paper"] = paper

    return state


def citation_node(state: ResearchState):

    citations = citation_agent.generate(
        state["search_results"],
        style=state.get("citation_style", "IEEE")
    )

    state["citations"] = citations

    return state


# -----------------------------
# Build Graph
# -----------------------------

workflow = StateGraph(ResearchState)

workflow.add_node("Search", search_node)
workflow.add_node("Parser", parser_node)
workflow.add_node("Chunk", chunk_node)
workflow.add_node("Embedding", embedding_node)
workflow.add_node("VectorStore", vectorstore_node)
workflow.add_node("Retrieval", retrieval_node)
workflow.add_node("Reranker", reranker_node)
workflow.add_node("Reviewer", reviewer_node)
workflow.add_node("Planner", planner_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Citation", citation_node)

workflow.add_edge(START, "Search")
workflow.add_edge("Search", "Parser")
workflow.add_edge("Parser", "Chunk")
workflow.add_edge("Chunk", "Embedding")
workflow.add_edge("Embedding", "VectorStore")
workflow.add_edge("VectorStore", "Retrieval")
workflow.add_edge("Retrieval", "Reranker")
workflow.add_edge("Reranker", "Reviewer")
workflow.add_edge("Reviewer", "Planner")
workflow.add_edge("Planner", "Writer")
workflow.add_edge("Writer", "Citation")
workflow.add_edge("Citation", END)

graph = workflow.compile()
