"""Unit tests for OpenRouter API client."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from backend.openrouter import query_model, query_models_parallel


@pytest.mark.asyncio
async def test_query_model_success():
    """Test successful model query."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response",
                    "reasoning_details": None,
                }
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        result = await query_model(
            "openai/gpt-4o", [{"role": "user", "content": "test"}]
        )

        assert result is not None
        assert result["content"] == "This is a test response"
        assert result["reasoning_details"] is None


@pytest.mark.asyncio
async def test_query_model_with_reasoning():
    """Test model query with reasoning details."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Response with reasoning",
                    "reasoning_details": {"tokens": 100, "steps": 5},
                }
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        result = await query_model(
            "openai/o1-preview", [{"role": "user", "content": "test"}]
        )

        assert result is not None
        assert result["content"] == "Response with reasoning"
        assert result["reasoning_details"] == {"tokens": 100, "steps": 5}


@pytest.mark.asyncio
async def test_query_model_http_error():
    """Test model query handles HTTP errors gracefully."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        result = await query_model(
            "openai/gpt-4o", [{"role": "user", "content": "test"}]
        )

        assert result is None


@pytest.mark.asyncio
async def test_query_model_timeout():
    """Test model query handles timeout gracefully."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("Timeout")
        )

        result = await query_model(
            "openai/gpt-4o", [{"role": "user", "content": "test"}], timeout=10.0
        )

        assert result is None


@pytest.mark.asyncio
async def test_query_model_malformed_response():
    """Test model query handles malformed response gracefully."""
    mock_response = Mock()
    mock_response.json.return_value = {"invalid": "structure"}
    mock_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        result = await query_model(
            "openai/gpt-4o", [{"role": "user", "content": "test"}]
        )

        assert result is None


@pytest.mark.asyncio
async def test_query_models_parallel_success():
    """Test parallel querying of multiple models."""
    models = ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"]

    with patch("backend.openrouter.query_model") as mock_query:
        mock_query.side_effect = [
            {"content": "Response from GPT-4", "reasoning_details": None},
            {"content": "Response from Claude", "reasoning_details": None},
            {"content": "Response from Gemini", "reasoning_details": None},
        ]

        results = await query_models_parallel(
            models, [{"role": "user", "content": "test"}]
        )

        assert len(results) == 3
        assert results["openai/gpt-4o"]["content"] == "Response from GPT-4"
        assert results["anthropic/claude-3-opus"]["content"] == "Response from Claude"
        assert results["google/gemini-pro"]["content"] == "Response from Gemini"


@pytest.mark.asyncio
async def test_query_models_parallel_partial_failure():
    """Test parallel querying handles partial failures gracefully."""
    models = ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"]

    with patch("backend.openrouter.query_model") as mock_query:
        mock_query.side_effect = [
            {"content": "Response from GPT-4", "reasoning_details": None},
            None,  # Claude fails
            {"content": "Response from Gemini", "reasoning_details": None},
        ]

        results = await query_models_parallel(
            models, [{"role": "user", "content": "test"}]
        )

        assert len(results) == 3
        assert results["openai/gpt-4o"]["content"] == "Response from GPT-4"
        assert results["anthropic/claude-3-opus"] is None
        assert results["google/gemini-pro"]["content"] == "Response from Gemini"


@pytest.mark.asyncio
async def test_query_models_parallel_all_fail():
    """Test parallel querying when all models fail."""
    models = ["openai/gpt-4o", "anthropic/claude-3-opus"]

    with patch("backend.openrouter.query_model") as mock_query:
        mock_query.side_effect = [None, None]

        results = await query_models_parallel(
            models, [{"role": "user", "content": "test"}]
        )

        assert len(results) == 2
        assert results["openai/gpt-4o"] is None
        assert results["anthropic/claude-3-opus"] is None


@pytest.mark.asyncio
async def test_query_models_parallel_empty_list():
    """Test parallel querying with empty model list."""
    results = await query_models_parallel([], [{"role": "user", "content": "test"}])

    assert len(results) == 0
    assert results == {}
