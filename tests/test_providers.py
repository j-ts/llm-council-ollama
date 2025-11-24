import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.providers import ProviderFactory, OpenRouterProvider, OllamaProvider, OpenAIProvider

@pytest.mark.asyncio
async def test_provider_factory():
    # Test OpenRouter default
    config = {"provider": "openrouter", "openrouter_api_key": "test"}
    provider = ProviderFactory.get_provider(config)
    assert isinstance(provider, OpenRouterProvider)
    assert provider.api_key == "test"

    # Test Ollama
    config = {"provider": "ollama", "ollama_base_url": "http://localhost:11434"}
    provider = ProviderFactory.get_provider(config)
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"

    # Test OpenAI
    config = {"provider": "openai", "openai_api_key": "test", "openai_base_url": "https://api.openai.com/v1"}
    provider = ProviderFactory.get_provider(config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == "test"
    assert provider.base_url == "https://api.openai.com/v1"

@pytest.mark.asyncio
async def test_ollama_provider_query():
    provider = OllamaProvider()
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Hello"},
        "prompt_eval_count": 10,
        "eval_count": 20
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        response = await provider.query("llama2", [{"role": "user", "content": "Hi"}])
        
        assert response["content"] == "Hello"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["cost"] == 0.0

@pytest.mark.asyncio
async def test_openai_provider_query():
    provider = OpenAIProvider(api_key="test")
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        response = await provider.query("gpt-3.5-turbo", [{"role": "user", "content": "Hi"}])
        
        assert response["content"] == "Hello"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
