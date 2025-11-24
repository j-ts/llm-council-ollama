"""
Model providers for LLM Council.
"""

import abc
import os
import httpx
from typing import List, Dict, Any, Optional
from .openrouter import query_model as openrouter_query_model

class ModelProvider(abc.ABC):
    """Abstract base class for model providers."""

    @abc.abstractmethod
    async def query(self, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """Query a model."""
        pass

    @abc.abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass

class OpenRouterProvider(ModelProvider):
    """Provider for OpenRouter."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # We might need to inject the key into the openrouter module or pass it explicitly
        # For now, we'll assume the openrouter module uses the global config, 
        # but ideally we should refactor openrouter.py to accept an api_key.
        # Since I haven't refactored openrouter.py yet, I will do that next. 
        pass

    async def query(self, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        return await openrouter_query_model(model, messages, timeout, api_key=self.api_key)

    async def list_models(self) -> List[str]:
        # OpenRouter has too many models to list effectively in a simple dropdown without search.
        # For now, we might return a curated list or allow free text.
        # The current app uses a hardcoded list in config.py.
        return [] 

class OllamaProvider(ModelProvider):
    """Provider for local Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def query(self, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Map Ollama response to the expected format
                return {
                    "content": data.get("message", {}).get("content", ""),
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        "cost": 0.0, # Local models are free
                        "cost_status": "final"
                    }
                }
        except Exception as e:
            print(f"Error querying Ollama model {model}: {e}")
            return None

    async def list_models(self) -> List[str]:
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
        return []

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI compatible APIs (Direct)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def query(self, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                message = data['choices'][0]['message']
                usage = data.get('usage', {})
                
                # Simple cost estimation could be added here if needed, but for now 0.
                return {
                    "content": message.get('content'),
                    "usage": {
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0),
                        "cost": 0.0, 
                        "cost_status": "unknown"
                    }
                }
        except Exception as e:
            print(f"Error querying OpenAI model {model}: {e}")
            return None

    async def list_models(self) -> List[str]:
        # Listing models from OpenAI compatible endpoints
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/models", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
        return []

class ProviderFactory:
    @staticmethod
    def get_provider(config: Dict[str, Any]) -> ModelProvider:
        provider_type = config.get("provider", "openrouter")
        
        if provider_type == "ollama":
            return OllamaProvider(base_url=config.get("ollama_base_url", "http://localhost:11434"))
        elif provider_type == "openai":
            return OpenAIProvider(
                api_key=config.get("openai_api_key", ""),
                base_url=config.get("openai_base_url", "https://api.openai.com/v1")
            )
        else:
            # Default to OpenRouter
            return OpenRouterProvider(api_key=config.get("openrouter_api_key", ""))
