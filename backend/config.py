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

    def _migrate_old_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old single-provider config to new multi-provider format."""
        
        # 1. Migrate from single-provider (root level) to multi-provider (providers dict)
        if "providers" not in old_config:
            print("Migrating configuration to new multi-provider format...")
            
            # Extract old provider info
            old_provider = old_config.get("provider", "openrouter")
            
            # Build new providers config
            providers = {
                "ollama": {
                    "base_url": old_config.get("ollama_base_url", "http://localhost:11434")
                },
                "openrouter": {
                    "api_key": old_config.get("openrouter_api_key", "")
                },
                "openai": [
                    {
                        "name": "Default",
                        "base_url": old_config.get("openai_base_url", "https://api.openai.com/v1"),
                        "api_key": old_config.get("openai_api_key", "")
                    }
                ]
            }
            
            # Convert council_models from list of strings to list of dicts
            old_council_models = old_config.get("council_models", [])
            council_models = []
            for model_name in old_council_models:
                council_models.append({
                    "name": model_name,
                    "provider": old_provider
                })
            
            # Convert chairman_model from string to dict
            old_chairman = old_config.get("chairman_model", "")
            chairman_model = {
                "name": old_chairman,
                "provider": old_provider
            }
            
            new_config = {
                "providers": providers,
                "council_models": council_models,
                "chairman_model": chairman_model
            }
            
            # Save the migrated config
            self._config = new_config
            self._save_config()
            print("Configuration migration complete!")
            return new_config

        # 2. Migrate from single OpenAI config (dict) to multiple OpenAI configs (list)
        # This handles the migration for users who are already on the multi-provider schema
        # but have the old "openai": {...} dict instead of "openai": [...] list
        if isinstance(old_config["providers"].get("openai"), dict):
            print("Migrating OpenAI config to multi-instance format...")
            old_openai = old_config["providers"]["openai"]
            old_config["providers"]["openai"] = [
                {
                    "name": "Default",
                    "base_url": old_openai.get("base_url", "https://api.openai.com/v1"),
                    "api_key": old_openai.get("api_key", "")
                }
            ]
            self._config = old_config
            self._save_config()
            print("OpenAI config migration complete!")
            return old_config

        return old_config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Migrate if necessary
                    return self._migrate_old_config(loaded_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Default configuration (new format)
        return {
            "providers": {
                "ollama": {
                    "base_url": "http://localhost:11434"
                },
                "openrouter": {
                    "api_key": OPENROUTER_API_KEY or ""
                },
                "openai": [
                    {
                        "name": "Default",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": ""
                    }
                ]
            },
            "council_models": [
                {"name": model, "provider": "openrouter"}
                for model in COUNCIL_MODELS
            ],
            "chairman_model": {
                "name": CHAIRMAN_MODEL,
                "provider": "openrouter"
            }
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
