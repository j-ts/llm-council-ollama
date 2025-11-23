"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b",
    "deepseek/deepseek-chat-v3-0324",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "mistralai/mistral-7b-instruct"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
