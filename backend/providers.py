"""
Model providers for LLM Council.

OpenRouter uses a dedicated provider so we can attach its billing headers and
usage polling, while Ollama (local) and OpenAI-compatible endpoints share the
lighter OpenAI-style interface.
"""

import abc
import json
import os
import httpx
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse
from .openrouter import query_model as openrouter_query_model

# Options that can be safely passed through to OpenAI-compatible chat completions
OPENAI_COMPATIBLE_OPTION_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "logit_bias",
    "seed",
    "response_format",
    "user",
}


def _filter_openai_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop unknown keys before forwarding to OpenAI-style providers."""
    if not isinstance(options, dict):
        return {}
    return {
        key: value
        for key, value in options.items()
        if key in OPENAI_COMPATIBLE_OPTION_KEYS and value is not None
    }


def _running_in_docker() -> bool:
    """Heuristic to detect container runtime for base URL rewrites."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            content = f.read()
            return "docker" in content or "containerd" in content
    except OSError:
        return False


def _rewrite_localhost_base_url(base_url: str) -> str:
    """
    Map localhost URLs to host.docker.internal when running in Docker,
    and back to 127.0.0.1 when running on the host but config contains host.docker.internal.
    """
    if not base_url:
        return base_url

    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port

    if not host:
        return base_url

    is_docker = _running_in_docker()
    new_host: Optional[str] = None

    if host in {"localhost", "127.0.0.1"} and is_docker:
        new_host = os.getenv("DOCKER_HOST_GATEWAY", "host.docker.internal")
    elif host in {"host.docker.internal", "gateway.docker.internal"} and not is_docker:
        new_host = "127.0.0.1"

    if new_host is None:
        return base_url

    netloc = new_host
    if port:
        netloc = f"{new_host}:{port}"

    return urlunparse(parsed._replace(netloc=netloc))


def _normalize_openai_base_url(base_url: str) -> str:
    """Ensure OpenAI-compatible base URLs include the expected /v1 prefix."""
    base_url = _rewrite_localhost_base_url(base_url)
    if not base_url:
        base_url = "https://api.openai.com/v1"
    parsed = urlparse(base_url)
    path = parsed.path or ""
    stripped_path = path.rstrip("/")

    if stripped_path == "":
        normalized_path = "/v1"
    elif stripped_path.endswith("/v1"):
        normalized_path = stripped_path
    else:
        normalized_path = f"{stripped_path}/v1"

    return urlunparse(parsed._replace(path=normalized_path))


class ModelProvider(abc.ABC):
    """Abstract base class for model providers."""

    @abc.abstractmethod
    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Query a model."""
        pass

    @abc.abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass

class OpenRouterProvider(ModelProvider):
    """Provider for OpenRouter (kept separate for billing headers and usage polling)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Kept distinct from generic OpenAI-compatible because of special headers/cost polling.

    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        safe_options = _filter_openai_options(options)
        return await openrouter_query_model(
            model,
            messages,
            timeout=timeout,
            api_key=self.api_key,
            extra_payload=safe_options if safe_options else None,
        )

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

    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        effective_options: Dict[str, Any] = {}
        # Provider defaults form the baseline; model-specific options override them.
        if self.options:
            effective_options.update({k: v for k, v in self.options.items() if v is not None})
        if options:
            effective_options.update({k: v for k, v in options.items() if v is not None})

        # Validate num_ctx early so we can return a user-friendly error.
        if "num_ctx" in effective_options:
            try:
                num_ctx_value = int(effective_options["num_ctx"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Ollama num_ctx must be a positive integer") from exc
            if num_ctx_value <= 0:
                raise ValueError("Ollama num_ctx must be a positive integer")
            effective_options["num_ctx"] = num_ctx_value

        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if effective_options:
            payload["options"] = effective_options
        
        try:
            timeout_obj = httpx.Timeout(timeout, connect=30.0)
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            async with httpx.AsyncClient(timeout=timeout_obj, limits=limits) as client:
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
        self.base_url = _normalize_openai_base_url(base_url)

    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        safe_options = _filter_openai_options(options)
        payload = {
            "model": model,
            "messages": messages
        }
        if safe_options:
            payload.update(safe_options)

        try:
            timeout_obj = httpx.Timeout(timeout, connect=30.0)
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            async with httpx.AsyncClient(timeout=timeout_obj, limits=limits) as client:
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
    def get_provider_from_model_config(model_config: Dict[str, Any], ollama_settings: Dict[str, Any] = None) -> ModelProvider:
        """
        Create a provider instance from a model registry entry.
        
        Args:
            model_config: Model configuration from registry with keys: type, model_name, base_url, api_key
            ollama_settings: Global Ollama settings (num_ctx, serialize_requests)
        
        Returns:
            Configured ModelProvider instance
        """
        model_type = model_config.get("type", "ollama")
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "")

        if model_type == "ollama":
            base_url = _rewrite_localhost_base_url(base_url or "http://localhost:11434")
        elif model_type == "openai-compatible":
            base_url = _normalize_openai_base_url(base_url or "https://api.openai.com/v1")
        
        # Create cache key based on model config
        cache_key = f"{model_type}::{base_url}::{api_key[:10] if api_key else ''}"
        
        if cache_key in ProviderFactory._provider_cache:
            return ProviderFactory._provider_cache[cache_key]
        
        # Create provider based on type
        if model_type == "ollama":
            num_ctx = None
            if ollama_settings:
                num_ctx = ollama_settings.get("num_ctx")
            provider = OllamaProvider(
                base_url=base_url,
                options={"num_ctx": num_ctx} if num_ctx else None
            )
        elif model_type == "openrouter":
            provider = OpenRouterProvider(api_key=api_key)
        elif model_type == "openai-compatible":
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        ProviderFactory._provider_cache[cache_key] = provider
        return provider
    
    @staticmethod
    def get_provider_for_model(model_config: Dict[str, Any], providers_config: Dict[str, Any]) -> ModelProvider:
        """
        LEGACY: Get the appropriate provider for a specific model configuration.
        This is kept for backward compatibility during migration.
        
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
                base_url=_normalize_openai_base_url(selected_config.get("base_url", "https://api.openai.com/v1"))
            )
            ProviderFactory._provider_cache[cache_key] = provider
            return provider

        # Standard handling for other providers
        provider_config = providers_config.get(provider_name, {})
        # Cache providers by base configuration; per-model options are applied at query-time.
        cache_key = f"{provider_name}:{json.dumps(provider_config, sort_keys=True)}"
        
        # Return cached provider if available
        if cache_key in ProviderFactory._provider_cache:
            return ProviderFactory._provider_cache[cache_key]
        
        # Create new provider instance
        if provider_name == "ollama":
            provider = OllamaProvider(
                base_url=_rewrite_localhost_base_url(provider_config.get("base_url", "http://localhost:11434")),
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
        LEGACY: Get all configured providers.
        
        Args:
            providers_config: Dict with provider configurations
        
        Returns:
            Dict mapping provider names/IDs to provider instances
        """
        providers = {}
        
        # Ollama
        if "ollama" in providers_config:
            providers["ollama"] = OllamaProvider(
                base_url=_rewrite_localhost_base_url(providers_config["ollama"].get("base_url", "http://localhost:11434")),
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
                        base_url=_normalize_openai_base_url(config.get("base_url", "https://api.openai.com/v1"))
                    )
        
        return providers
    
    @staticmethod
    def clear_cache():
        """Reset cached provider instances (used when credentials/configs change)."""
        ProviderFactory._provider_cache.clear()
