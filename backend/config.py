"""Configuration for the LLM Council."""

import os
import json
import uuid
import copy
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
DEFAULT_GENERAL_SETTINGS = {"use_env_for_api_keys": False}

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
        self._ensure_general_settings()

    def _ensure_general_settings(self):
        """Ensure general settings block exists with defaults."""
        general_settings = self._config.get("general_settings", {})
        merged_settings = {**DEFAULT_GENERAL_SETTINGS, **general_settings} if isinstance(general_settings, dict) else DEFAULT_GENERAL_SETTINGS.copy()
        self._config["general_settings"] = merged_settings

    def _migrate_old_to_registry(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old provider-based config to new model registry format."""
        
        # Check if already using new format
        if "models" in old_config and "ollama_settings" in old_config:
            return old_config
        
        print("Migrating to model registry format...")
        
        models = {}
        council_model_ids = []
        chairman_model_id = None
        
        # Migrate from providers format
        if "providers" in old_config:
            providers = old_config.get("providers", {})
            council_models_old = old_config.get("council_models", [])
            chairman_model_old = old_config.get("chairman_model", {})
            
            # Convert council models
            for model_entry in council_models_old:
                if isinstance(model_entry, str):
                    # Very old format
                    model_id = str(uuid.uuid4())
                    models[model_id] = {
                        "label": model_entry,
                        "type": "openrouter",
                        "model_name": model_entry,
                        "api_key": providers.get("openrouter", {}).get("api_key", "")
                    }
                    council_model_ids.append(model_id)
                elif isinstance(model_entry, dict):
                    model_id = str(uuid.uuid4())
                    provider = model_entry.get("provider", "ollama")
                    model_name = model_entry.get("name", "")
                    
                    if provider == "ollama":
                        models[model_id] = {
                            "label": model_name,
                            "type": "ollama",
                            "model_name": model_name,
                            "base_url": providers.get("ollama", {}).get("base_url", "http://localhost:11434")
                        }
                    elif provider == "openrouter":
                        models[model_id] = {
                            "label": model_name,
                            "type": "openrouter",
                            "model_name": model_name,
                            "api_key": providers.get("openrouter", {}).get("api_key", "")
                        }
                    elif provider == "openai":
                        openai_configs = providers.get("openai", [])
                        if isinstance(openai_configs, dict):
                            openai_configs = [openai_configs]
                        config_name = model_entry.get("openai_config_name")
                        matching_config = None
                        if config_name:
                            matching_config = next((c for c in openai_configs if c.get("name") == config_name), None)
                        if not matching_config and openai_configs:
                            matching_config = openai_configs[0]
                        
                        if matching_config:
                            models[model_id] = {
                                "label": f"{model_name} ({matching_config.get('name', 'OpenAI')})",
                                "type": "openai-compatible",
                                "model_name": model_name,
                                "base_url": matching_config.get("base_url", "https://api.openai.com/v1"),
                                "api_key": matching_config.get("api_key", "")
                            }
                    
                    council_model_ids.append(model_id)
            
            # Convert chairman model
            if isinstance(chairman_model_old, str):
                chairman_model_id = str(uuid.uuid4())
                models[chairman_model_id] = {
                    "label": chairman_model_old,
                    "type": "openrouter",
                    "model_name": chairman_model_old,
                    "api_key": providers.get("openrouter", {}).get("api_key", "")
                }
            elif isinstance(chairman_model_old, dict):
                chairman_model_id = str(uuid.uuid4())
                provider = chairman_model_old.get("provider", "ollama")
                model_name = chairman_model_old.get("name", "")
                
                if provider == "ollama":
                    models[chairman_model_id] = {
                        "label": model_name,
                        "type": "ollama",
                        "model_name": model_name,
                        "base_url": providers.get("ollama", {}).get("base_url", "http://localhost:11434")
                    }
                elif provider == "openrouter":
                    models[chairman_model_id] = {
                        "label": model_name,
                        "type": "openrouter",
                        "model_name": model_name,
                        "api_key": providers.get("openrouter", {}).get("api_key", "")
                    }
                elif provider == "openai":
                    openai_configs = providers.get("openai", [])
                    if isinstance(openai_configs, dict):
                        openai_configs = [openai_configs]
                    config_name = chairman_model_old.get("openai_config_name")
                    matching_config = None
                    if config_name:
                        matching_config = next((c for c in openai_configs if c.get("name") == config_name), None)
                    if not matching_config and openai_configs:
                        matching_config = openai_configs[0]
                    
                    if matching_config:
                        models[chairman_model_id] = {
                            "label": f"{model_name} ({matching_config.get('name', 'OpenAI')})",
                            "type": "openai-compatible",
                            "model_name": model_name,
                            "base_url": matching_config.get("base_url", "https://api.openai.com/v1"),
                            "api_key": matching_config.get("api_key", "")
                        }
            
            # Extract ollama settings
            ollama_settings = {
                "num_ctx": providers.get("ollama", {}).get("num_ctx", 4096),
                "serialize_requests": old_config.get("serialize_local_models", False)
            }
        else:
            # Very old format or empty
            ollama_settings = {
                "num_ctx": 4096,
                "serialize_requests": False
            }
        
        new_config = {
            "models": models,
            "ollama_settings": ollama_settings,
            "council_models": council_model_ids,
            "chairman_model": chairman_model_id,
            "general_settings": DEFAULT_GENERAL_SETTINGS.copy()
        }
        
        # Save migrated config
        self._config = new_config
        self._save_config()
        print("Migration to model registry complete!")
        return new_config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Migrate if necessary
                    migrated_config = self._migrate_old_to_registry(loaded_config)
                    if not isinstance(migrated_config.get("general_settings"), dict):
                        migrated_config["general_settings"] = DEFAULT_GENERAL_SETTINGS.copy()
                    else:
                        migrated_config["general_settings"] = {
                            **DEFAULT_GENERAL_SETTINGS,
                            **migrated_config.get("general_settings", {})
                        }
                    return migrated_config
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Default configuration (new model registry format)
        return {
            "models": {},
            "ollama_settings": {
                "num_ctx": 4096,
                "serialize_requests": False
            },
            "general_settings": DEFAULT_GENERAL_SETTINGS.copy(),
            "council_models": [],
            "chairman_model": None
        }

    @staticmethod
    def _mask_api_key(api_key: Any) -> str:
        """Mask API keys while keeping env references visible."""
        if not isinstance(api_key, str):
            return ""
        if api_key.startswith("env:"):
            return api_key
        return "***" if api_key else ""

    def _masked_config(self) -> Dict[str, Any]:
        """Return a deep copy of the config with API keys masked."""
        masked = copy.deepcopy(self._config)

        # Mask model registry entries
        models = masked.get("models", {})
        if isinstance(models, dict):
            for model in models.values():
                if isinstance(model, dict) and "api_key" in model:
                    model["api_key"] = self._mask_api_key(model.get("api_key", ""))

        # Mask legacy provider configs if present
        providers = masked.get("providers", {})
        if isinstance(providers, dict):
            if "openrouter" in providers and isinstance(providers.get("openrouter"), dict):
                providers["openrouter"]["api_key"] = self._mask_api_key(providers["openrouter"].get("api_key", ""))

            openai_configs = providers.get("openai")
            if isinstance(openai_configs, dict):
                openai_configs = [openai_configs]
                providers["openai"] = openai_configs
            if isinstance(openai_configs, list):
                for cfg in openai_configs:
                    if isinstance(cfg, dict) and "api_key" in cfg:
                        cfg["api_key"] = self._mask_api_key(cfg.get("api_key", ""))

        return masked

    def get_config(self, mask_sensitive: bool = False) -> Dict[str, Any]:
        """Get current configuration."""
        self._ensure_general_settings()
        if mask_sensitive:
            return self._masked_config()
        return self._config

    @staticmethod
    def _merge_api_key(existing: Any, incoming: Any) -> str:
        """Merge API key values while honoring masked placeholders."""
        if incoming == "***":
            return existing or ""
        return incoming or ""

    def _merge_models(self, existing_models: Dict[str, Any], incoming_models: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model registry entries, preserving secrets when masked."""
        merged_models: Dict[str, Any] = {}
        for model_id, incoming_model in (incoming_models or {}).items():
            existing_model = existing_models.get(model_id, {})
            updated_model = {**existing_model, **(incoming_model or {})}

            if "api_key" in incoming_model:
                updated_model["api_key"] = self._merge_api_key(existing_model.get("api_key", ""), incoming_model.get("api_key", ""))
            elif "api_key" in existing_model:
                updated_model["api_key"] = existing_model.get("api_key", "")

            merged_models[model_id] = updated_model

        return merged_models

    def _merge_openai_providers(self, existing: Any, incoming: Any) -> List[Dict[str, Any]]:
        """Merge OpenAI provider list/dict formats while preserving secrets."""
        existing_list = existing
        if isinstance(existing_list, dict):
            existing_list = [existing_list]
        if not isinstance(existing_list, list):
            existing_list = []

        incoming_list = incoming
        if isinstance(incoming_list, dict):
            incoming_list = [incoming_list]
        if not isinstance(incoming_list, list):
            incoming_list = []

        merged_list = []
        for idx, cfg in enumerate(incoming_list):
            base_cfg = existing_list[idx] if idx < len(existing_list) else {}
            updated_cfg = {**base_cfg, **(cfg or {})}
            if isinstance(cfg, dict) and "api_key" in cfg:
                updated_cfg["api_key"] = self._merge_api_key(base_cfg.get("api_key", ""), cfg.get("api_key", ""))
            elif "api_key" in base_cfg:
                updated_cfg["api_key"] = base_cfg.get("api_key", "")
            merged_list.append(updated_cfg)

        return merged_list

    def _merge_providers(self, existing_providers: Dict[str, Any], incoming_providers: Dict[str, Any]) -> Dict[str, Any]:
        """Merge legacy provider block, preserving secrets."""
        merged_providers = copy.deepcopy(existing_providers) if isinstance(existing_providers, dict) else {}
        for name, provider_cfg in (incoming_providers or {}).items():
            if name == "openai":
                existing_openai = existing_providers.get("openai") if isinstance(existing_providers, dict) else None
                merged_providers[name] = self._merge_openai_providers(existing_openai, provider_cfg)
                continue

            if isinstance(provider_cfg, dict):
                base_cfg = existing_providers.get(name, {}) if isinstance(existing_providers, dict) else {}
                updated_cfg = {**base_cfg, **provider_cfg}
                if "api_key" in provider_cfg:
                    updated_cfg["api_key"] = self._merge_api_key(base_cfg.get("api_key", ""), provider_cfg.get("api_key", ""))
                elif "api_key" in base_cfg:
                    updated_cfg["api_key"] = base_cfg.get("api_key", "")
                merged_providers[name] = updated_cfg
            else:
                merged_providers[name] = provider_cfg

        return merged_providers

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration and save to file."""
        merged_config = copy.deepcopy(self._config)

        # General settings (add defaults)
        if "general_settings" in new_config:
            merged_config["general_settings"] = {
                **DEFAULT_GENERAL_SETTINGS,
                **merged_config.get("general_settings", {}),
                **(new_config.get("general_settings") or {})
            }
        else:
            merged_config["general_settings"] = {**DEFAULT_GENERAL_SETTINGS, **merged_config.get("general_settings", {})}

        # Models
        if "models" in new_config:
            merged_config["models"] = self._merge_models(merged_config.get("models", {}), new_config.get("models", {}))

        # Legacy providers
        if "providers" in new_config:
            merged_config["providers"] = self._merge_providers(merged_config.get("providers", {}), new_config.get("providers", {}))

        # Ollama settings and other top-level keys
        if "ollama_settings" in new_config:
            merged_config["ollama_settings"] = {
                **merged_config.get("ollama_settings", {}),
                **(new_config.get("ollama_settings") or {})
            }

        for key in ["council_models", "chairman_model", "serialize_local_models"]:
            if key in new_config:
                merged_config[key] = new_config[key]

        self._config = merged_config
        self._save_config()
        return self._config

    def _save_config(self):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)

# Global config manager instance
config_manager = ConfigManager()
