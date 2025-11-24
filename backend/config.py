"""Configuration for the LLM Council."""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key (Legacy/Default)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers (Legacy/Default)
COUNCIL_MODELS = [
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b",
    "deepseek/deepseek-chat-v3-0324",
]

# Chairman model - synthesizes final response (Legacy/Default)
CHAIRMAN_MODEL = "mistralai/mistral-7b-instruct"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
CONFIG_FILE = "data/config.json"

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

class ConfigManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Default configuration
        return {
            "provider": "openrouter",
            "openrouter_api_key": OPENROUTER_API_KEY,
            "ollama_base_url": "http://localhost:11434",
            "openai_api_key": "",
            "openai_base_url": "https://api.openai.com/v1",
            "council_models": COUNCIL_MODELS,
            "chairman_model": CHAIRMAN_MODEL
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration and save to file."""
        self._config.update(new_config)
        self._save_config()
        return self._config

    def _save_config(self):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)

# Global config manager instance
config_manager = ConfigManager()
