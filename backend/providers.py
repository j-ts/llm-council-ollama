"""
Model providers for LLM Council.
"""

import abc
import os
import json
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

    def __init__(self, base_url: str = "http://localhost:11434", options: Optional[Dict[str, Any]] = None):
        self.base_url = base_url.rstrip("/")
        # Optional per-request options (e.g., num_ctx) to avoid overallocating KV cache
        self.options = options or {}

    async def query(self, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if self.options:
            payload["options"] = {k: v for k, v in self.options.items() if v is not None}
        
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
        except httpx.HTTPStatusError as e:
            detail = self._extract_error_detail(e.response)
            if e.response and e.response.status_code == 404:
                detail = f"{detail} (pull the model with 'ollama pull {model}')"
            raise RuntimeError(
                f"Ollama error ({e.response.status_code if e.response else 'n/a'}) for model '{model}': {detail}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Unable to reach Ollama at {self.base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama request failed for model '{model}': {e}") from e

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

    @staticmethod
    def _extract_error_detail(response: Optional[httpx.Response]) -> str:
        """Best-effort extraction of error detail from Ollama."""
        if response is None:
            return "Unknown error"
        try:
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                return data["error"]
        except ValueError:
            pass
        text = response.text.strip()
        return text or "Unknown error"

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
        except httpx.HTTPStatusError as e:
            detail = self._extract_error_detail(e.response)
            raise RuntimeError(
                f"OpenAI-compatible error ({e.response.status_code if e.response else 'n/a'}) for model '{model}': {detail}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Unable to reach OpenAI-compatible endpoint at {self.base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible request failed for model '{model}': {e}") from e

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

    @staticmethod
    def _extract_error_detail(response: Optional[httpx.Response]) -> str:
        """Best-effort extraction of error detail from OpenAI compatible APIs."""
        if response is None:
            return "Unknown error"
        try:
            data = response.json()
            if isinstance(data, dict):
                if data.get("error"):
                    err = data["error"]
                    if isinstance(err, dict):
                        return err.get("message") or str(err)
                    return str(err)
        except ValueError:
            pass
        text = response.text.strip()
        return text or "Unknown error"

class ProviderFactory:
    _provider_cache: Dict[str, ModelProvider] = {}
    
    @staticmethod
    def get_provider_for_model(model_config: Dict[str, Any], providers_config: Dict[str, Any]) -> ModelProvider:
        """
        Get the appropriate provider for a specific model configuration.
        
        Args:
            model_config: Dict with 'name', 'provider', and optional 'openai_config_name' keys
            providers_config: Dict with provider configurations
        
        Returns:
            Configured ModelProvider instance
        """
        provider_name = model_config.get("provider")
        
        # Handle OpenAI specifically as it can have multiple configurations
        if provider_name == "openai":
            openai_configs = providers_config.get("openai", [])
            # If it's a dict (legacy/migration edge case), wrap in list
            if isinstance(openai_configs, dict):
                openai_configs = [openai_configs]
            
            # Find the specific config
            target_config_name = model_config.get("openai_config_name")
            selected_config = None
            
            if target_config_name:
                for config in openai_configs:
                    if config.get("name") == target_config_name:
                        selected_config = config
                        break
            
            # Default to first config if not found or not specified
            if not selected_config and openai_configs:
                selected_config = openai_configs[0]
            
            if not selected_config:
                # Fallback if no config exists at all
                selected_config = {"base_url": "https://api.openai.com/v1", "api_key": ""}
            
            cache_key = f"openai:{selected_config.get('name', 'default')}:{selected_config.get('base_url')}"
            
            if cache_key in ProviderFactory._provider_cache:
                return ProviderFactory._provider_cache[cache_key]
                
            provider = OpenAIProvider(
                api_key=selected_config.get("api_key", ""),
                base_url=selected_config.get("base_url", "https://api.openai.com/v1")
            )
            ProviderFactory._provider_cache[cache_key] = provider
            return provider

        # Standard handling for other providers
        provider_config = providers_config.get(provider_name, {})
        cache_key = f"{provider_name}:{json.dumps(provider_config, sort_keys=True)}"
        
        # Return cached provider if available
        if cache_key in ProviderFactory._provider_cache:
            return ProviderFactory._provider_cache[cache_key]
        
        # Create new provider instance
        if provider_name == "ollama":
            provider = OllamaProvider(
                base_url=provider_config.get("base_url", "http://localhost:11434"),
                options={"num_ctx": provider_config.get("num_ctx")}
            )
        elif provider_name == "openrouter":
            provider = OpenRouterProvider(api_key=provider_config.get("api_key", ""))
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Cache the provider
        ProviderFactory._provider_cache[cache_key] = provider
        return provider
    
    @staticmethod
    def get_all_providers(providers_config: Dict[str, Any]) -> Dict[str, ModelProvider]:
        """
        Get all configured providers.
        
        Args:
            providers_config: Dict with provider configurations
        
        Returns:
            Dict mapping provider names/IDs to provider instances
        """
        providers = {}
        
        # Ollama
        if "ollama" in providers_config:
            providers["ollama"] = OllamaProvider(
                base_url=providers_config["ollama"].get("base_url", "http://localhost:11434"),
                options={"num_ctx": providers_config["ollama"].get("num_ctx")}
            )
        
        # OpenRouter
        if "openrouter" in providers_config and providers_config["openrouter"].get("api_key"):
            providers["openrouter"] = OpenRouterProvider(
                api_key=providers_config["openrouter"]["api_key"]
            )
        
        # OpenAI (Multiple instances)
        if "openai" in providers_config:
            openai_configs = providers_config["openai"]
            if isinstance(openai_configs, dict):
                openai_configs = [openai_configs]
                
            for config in openai_configs:
                if config.get("api_key"):
                    name = config.get("name", "Default")
                    # Use a composite key for the frontend to identify specific OpenAI instances
                    # Format: openai:{name}
                    key = f"openai:{name}"
                    providers[key] = OpenAIProvider(
                        api_key=config["api_key"],
                        base_url=config.get("base_url", "https://api.openai.com/v1")
                    )
        
        return providers
    
    @staticmethod
    def get_provider(config: Dict[str, Any]) -> ModelProvider:
        """
        Legacy method for backward compatibility.
        Gets a single provider based on old config format.
        """
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

    @staticmethod
    def clear_cache():
        """Reset cached provider instances (used when credentials/configs change)."""
        ProviderFactory._provider_cache.clear()
