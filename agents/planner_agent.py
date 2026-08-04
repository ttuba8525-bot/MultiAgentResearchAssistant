from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.helpers import get_groq_api_key


class PlannerAgent:
    """
    Generates novel research ideas based on
    identified research gaps.
    """

    def __init__(self):

        self.llm = ChatGroq(
            api_key=get_groq_api_key(),
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are a senior AI researcher.

Research Topic:
{query}

Research Gaps:
{review}

Based on these gaps, generate:

1. Novel Research Idea
2. Research Objectives
3. Proposed Methodology
4. Expected Contributions
5. Potential Challenges
6. Future Extensions

The idea should be original, practical, and suitable for publication.

Return the response in Markdown.
"""
        )

        self.chain = self.prompt | self.llm

    def generate_plan(self, query, review):

        response = self.chain.invoke(
            {
                "query": query,
                "review": review
            }
        )

        return response.content
