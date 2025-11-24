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

# Redis configuration for background jobs
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# Model pricing (USD per 1M tokens) - Updated from OpenRouter
# Used for estimated cost calculation (actual cost fetched from generation API)
MODEL_PRICING = {
    # Pricing from OpenRouter API (updated 2025-11-24)
    "openai/gpt-5.1": {"input": 1.25, "output": 10.00},
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "x-ai/grok-4": {"input": 3.00, "output": 15.00},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},  # Used for title generation only
}

DEFAULT_PRICING = {"input": 1.00, "output": 3.00}  # Fallback for unknown models

def calculate_estimated_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate estimated cost based on token usage.
    This is an approximation - actual cost will be fetched from OpenRouter generation API.
    
    Args:
        model: Model identifier
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
