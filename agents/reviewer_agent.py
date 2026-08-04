from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.helpers import get_groq_api_key


class ReviewerAgent:
    """
    Reviews retrieved papers and generates
    literature review, comparison and research gaps.
    """

    def __init__(self):

        self.llm = ChatGroq(
            api_key=get_groq_api_key(),
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert academic researcher.

Given the following research paper excerpts:

{context}

Research Topic:
{query}

Generate:

1. Literature Review
2. Comparison of Existing Methods
3. Strengths
4. Limitations
5. Research Gaps
6. Future Work

Return the answer in well-structured Markdown.
"""
        )

        self.chain = self.prompt | self.llm

    def review(self, query, documents):

        context = "\n\n".join(
            [
                doc["text"]
                for doc in documents
            ]
        )

        response = self.chain.invoke(
            {
                "query": query,
                "context": context
            }
        )

        return response.content
