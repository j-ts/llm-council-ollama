"""Unit tests for council orchestration logic."""

import pytest
from unittest.mock import AsyncMock, patch
from backend.council import (
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    parse_ranking_from_text,
    calculate_aggregate_rankings,
    generate_conversation_title,
    run_full_council,
)


@pytest.mark.asyncio
async def test_stage1_collect_responses_success():
    """Test Stage 1 collects responses from all models."""
    mock_responses = {
        "openai/gpt-4o": {"content": "GPT response", "reasoning_details": None},
        "anthropic/claude-3-opus": {
            "content": "Claude response",
            "reasoning_details": None,
        },
    }

    with patch("backend.council.query_models_parallel", new=AsyncMock()) as mock_query:
        mock_query.return_value = mock_responses

        results = await stage1_collect_responses("What is AI?")

        assert len(results) == 2
        assert results[0]["model"] == "openai/gpt-4o"
        assert results[0]["response"] == "GPT response"
        assert results[1]["model"] == "anthropic/claude-3-opus"
        assert results[1]["response"] == "Claude response"


@pytest.mark.asyncio
async def test_stage1_filters_failed_responses():
    """Test Stage 1 filters out failed model responses."""
    mock_responses = {
        "openai/gpt-4o": {"content": "GPT response", "reasoning_details": None},
        "anthropic/claude-3-opus": None,  # Failed
        "google/gemini-pro": {"content": "Gemini response", "reasoning_details": None},
    }

    with patch("backend.council.query_models_parallel", new=AsyncMock()) as mock_query:
        mock_query.return_value = mock_responses

        results = await stage1_collect_responses("What is AI?")

        assert len(results) == 2
        assert all(r["response"] for r in results)


@pytest.mark.asyncio
async def test_stage2_collect_rankings_anonymizes():
    """Test Stage 2 anonymizes responses for ranking."""
    stage1_results = [
        {"model": "openai/gpt-4o", "response": "Response 1"},
        {"model": "anthropic/claude-3-opus", "response": "Response 2"},
    ]

    mock_responses = {
        "openai/gpt-4o": {
            "content": "FINAL RANKING:\n1. Response B\n2. Response A",
            "reasoning_details": None,
        },
        "anthropic/claude-3-opus": {
            "content": "FINAL RANKING:\n1. Response A\n2. Response B",
            "reasoning_details": None,
        },
    }

    with patch("backend.council.query_models_parallel", new=AsyncMock()) as mock_query:
        mock_query.return_value = mock_responses

        results, label_to_model = await stage2_collect_rankings(
            "What is AI?", stage1_results
        )

        assert len(results) == 2
        assert "Response A" in label_to_model
        assert "Response B" in label_to_model
        assert label_to_model["Response A"] == "openai/gpt-4o"
        assert label_to_model["Response B"] == "anthropic/claude-3-opus"


@pytest.mark.asyncio
async def test_stage3_synthesize_final():
    """Test Stage 3 synthesizes final response."""
    stage1_results = [
        {"model": "openai/gpt-4o", "response": "GPT response"},
    ]

    stage2_results = [
        {
            "model": "openai/gpt-4o",
            "ranking": "FINAL RANKING:\n1. Response A",
            "parsed_ranking": ["Response A"],
        }
    ]

    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = {
            "content": "Final synthesized answer",
            "reasoning_details": None,
        }

        result = await stage3_synthesize_final(
            "What is AI?", stage1_results, stage2_results
        )

        assert result["model"] == "google/gemini-3-pro-preview"
        assert result["response"] == "Final synthesized answer"


@pytest.mark.asyncio
async def test_stage3_handles_chairman_failure():
    """Test Stage 3 returns error message if chairman fails."""
    stage1_results = [{"model": "openai/gpt-4o", "response": "GPT response"}]
    stage2_results = [
        {
            "model": "openai/gpt-4o",
            "ranking": "FINAL RANKING:\n1. Response A",
            "parsed_ranking": ["Response A"],
        }
    ]

    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = None

        result = await stage3_synthesize_final(
            "What is AI?", stage1_results, stage2_results
        )

        assert "Error" in result["response"]


def test_parse_ranking_from_text_numbered_format():
    """Test parsing numbered ranking format."""
    text = """
    Response A is good but lacks depth.
    Response B is excellent.
    Response C is mediocre.

    FINAL RANKING:
    1. Response B
    2. Response A
    3. Response C
    """

    result = parse_ranking_from_text(text)

    assert result == ["Response B", "Response A", "Response C"]


def test_parse_ranking_from_text_without_numbers():
    """Test parsing ranking without numbers."""
    text = """
    Response A is good.
    Response B is better.

    FINAL RANKING:
    Response B
    Response A
    """

    result = parse_ranking_from_text(text)

    assert result == ["Response B", "Response A"]


def test_parse_ranking_from_text_no_final_ranking_header():
    """Test parsing when no FINAL RANKING header exists."""
    text = """
    Response A is mentioned first.
    Response B is mentioned second.
    """

    result = parse_ranking_from_text(text)

    # Should still extract Response labels in order
    assert result == ["Response A", "Response B"]


def test_parse_ranking_from_text_empty():
    """Test parsing empty text."""
    result = parse_ranking_from_text("")
    assert result == []


def test_calculate_aggregate_rankings():
    """Test calculating aggregate rankings across all models."""
    stage2_results = [
        {
            "model": "model1",
            "ranking": "FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C",
            "parsed_ranking": ["Response A", "Response B", "Response C"],
        },
        {
            "model": "model2",
            "ranking": "FINAL RANKING:\n1. Response B\n2. Response A\n3. Response C",
            "parsed_ranking": ["Response B", "Response A", "Response C"],
        },
        {
            "model": "model3",
            "ranking": "FINAL RANKING:\n1. Response A\n2. Response C\n3. Response B",
            "parsed_ranking": ["Response A", "Response C", "Response B"],
        },
    ]

    label_to_model = {
        "Response A": "openai/gpt-4o",
        "Response B": "anthropic/claude-3-opus",
        "Response C": "google/gemini-pro",
    }

    result = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Response A: positions [1, 2, 1] -> avg 1.33
    # Response B: positions [2, 1, 3] -> avg 2.0
    # Response C: positions [3, 3, 2] -> avg 2.67

    assert len(result) == 3
    assert result[0]["model"] == "openai/gpt-4o"  # Best average rank
    assert result[0]["average_rank"] == 1.33
    assert result[0]["rankings_count"] == 3

    assert result[1]["model"] == "anthropic/claude-3-opus"
    assert result[1]["average_rank"] == 2.0

    assert result[2]["model"] == "google/gemini-pro"  # Worst average rank
    assert result[2]["average_rank"] == 2.67


def test_calculate_aggregate_rankings_with_missing_labels():
    """Test aggregate rankings when some models don't rank all responses."""
    stage2_results = [
        {
            "model": "model1",
            "ranking": "FINAL RANKING:\n1. Response A\n2. Response B",
            "parsed_ranking": ["Response A", "Response B"],
        },
        {
            "model": "model2",
            "ranking": "FINAL RANKING:\n1. Response B",
            "parsed_ranking": ["Response B"],
        },
    ]

    label_to_model = {
        "Response A": "openai/gpt-4o",
        "Response B": "anthropic/claude-3-opus",
    }

    result = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Response A: [1] from 1 model
    # Response B: [2, 1] from 2 models

    assert len(result) == 2
    assert result[0]["model"] == "openai/gpt-4o"
    assert result[0]["rankings_count"] == 1

    assert result[1]["model"] == "anthropic/claude-3-opus"
    assert result[1]["rankings_count"] == 2


@pytest.mark.asyncio
async def test_generate_conversation_title_success():
    """Test generating conversation title."""
    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = {
            "content": "Understanding Artificial Intelligence",
            "reasoning_details": None,
        }

        title = await generate_conversation_title("What is AI?")

        assert title == "Understanding Artificial Intelligence"


@pytest.mark.asyncio
async def test_generate_conversation_title_fallback():
    """Test generating conversation title falls back on failure."""
    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = None

        title = await generate_conversation_title("What is AI?")

        assert title == "New Conversation"


@pytest.mark.asyncio
async def test_generate_conversation_title_strips_quotes():
    """Test title generation strips quotes."""
    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = {
            "content": '"Understanding AI"',
            "reasoning_details": None,
        }

        title = await generate_conversation_title("What is AI?")

        assert title == "Understanding AI"


@pytest.mark.asyncio
async def test_generate_conversation_title_truncates_long():
    """Test title generation truncates long titles."""
    with patch("backend.council.query_model", new=AsyncMock()) as mock_query:
        mock_query.return_value = {
            "content": "A" * 60,  # Very long title
            "reasoning_details": None,
        }

        title = await generate_conversation_title("What is AI?")

        assert len(title) == 50
        assert title.endswith("...")


@pytest.mark.asyncio
async def test_run_full_council_success():
    """Test full council process end-to-end."""
    with (
        patch("backend.council.stage1_collect_responses", new=AsyncMock()) as mock_s1,
        patch("backend.council.stage2_collect_rankings", new=AsyncMock()) as mock_s2,
        patch("backend.council.stage3_synthesize_final", new=AsyncMock()) as mock_s3,
    ):
        # Setup mock responses
        mock_s1.return_value = [
            {"model": "model1", "response": "Response 1"},
            {"model": "model2", "response": "Response 2"},
        ]

        mock_s2.return_value = (
            [
                {
                    "model": "model1",
                    "ranking": "FINAL RANKING:\n1. Response A\n2. Response B",
                    "parsed_ranking": ["Response A", "Response B"],
                }
            ],
            {"Response A": "model1", "Response B": "model2"},
        )

        mock_s3.return_value = {"model": "chairman", "response": "Final answer"}

        stage1, stage2, stage3, metadata = await run_full_council("What is AI?")

        assert len(stage1) == 2
        assert len(stage2) == 1
        assert stage3["response"] == "Final answer"
        assert "label_to_model" in metadata
        assert "aggregate_rankings" in metadata


@pytest.mark.asyncio
async def test_run_full_council_all_models_fail():
    """Test full council when all models fail in Stage 1."""
    with patch("backend.council.stage1_collect_responses", new=AsyncMock()) as mock_s1:
        mock_s1.return_value = []  # No successful responses

        stage1, stage2, stage3, metadata = await run_full_council("What is AI?")

        assert len(stage1) == 0
        assert len(stage2) == 0
        assert "failed" in stage3["response"].lower()
        assert stage3["model"] == "error"
