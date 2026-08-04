import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()


def get_groq_api_key():
    """
    Returns the GROQ API Key.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.stop()

    return api_key


def get_tavily_api_key():
    """
    Returns the Tavily API Key.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        st.error("❌ TAVILY_API_KEY not found in .env file")
        st.stop()

    return api_key


def ensure_directory(path: str):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_project_directories():
    """
    Create all required folders automatically.
    """

    folders = [
        "data",
        "data/papers",
        "data/vector_db",
        "data/output"
    ]

    for folder in folders:
        ensure_directory(folder)


def success_message(msg):
    st.success(msg)


def info_message(msg):
    st.info(msg)


def warning_message(msg):
    st.warning(msg)


def error_message(msg):
    st.error(msg)
