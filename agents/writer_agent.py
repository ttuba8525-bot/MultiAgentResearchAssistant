from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.helpers import get_groq_api_key


class WriterAgent:
    """
    Generates a complete IEEE-style research paper.
    """

    def __init__(self):

        self.llm = ChatGroq(
            api_key=get_groq_api_key(),
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert academic writer.

Research Topic:
{query}

Literature Review:
{review}

Novel Research Plan:
{plan}

Write a complete IEEE-style research paper.

Include the following sections:

# Title

# Abstract

# Keywords

# 1. Introduction

# 2. Literature Review

# 3. Proposed Methodology

# 4. Experimental Setup

# 5. Expected Results

# 6. Future Work

# 7. Conclusion

Return the paper in Markdown.
"""
        )

        self.chain = self.prompt | self.llm

    def write_paper(self, query, review, plan):

        response = self.chain.invoke(
            {
                "query": query,
                "review": review,
                "plan": plan
            }
        )

        return response.content
