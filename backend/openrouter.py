"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        api_key: Optional API key override

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    key = api_key or OPENROUTER_API_KEY
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "include_usage": True  # Enable usage accounting to get cost data
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']
            
            # Extract usage and cost information from OpenRouter API
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            # Calculate ESTIMATED cost from tokens (immediate)
            # Actual cost will be fetched later from generation API
            from .config import calculate_estimated_cost
            estimated_cost = calculate_estimated_cost(model, prompt_tokens, completion_tokens)
            
            # Extract generation ID for later cost polling
            generation_id = data.get('id', None)
            
            cost_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': usage.get('total_tokens', 0),
                'cost': estimated_cost,  # Estimated cost (immediate)
                'cost_status': 'estimated',  # Mark as estimate
                'generation_id': generation_id  # For polling actual cost later
            }

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'usage': cost_info
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        api_key: Optional API key override

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages, api_key=api_key) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def fetch_actual_cost(generation_id: str) -> float:
    """
    Fetch actual cost from OpenRouter generation API.
    This polls the /generation endpoint to get the real cost after calculation.
    
    Args:
        generation_id: The generation ID from the chat completion response
    
    Returns:
        Actual cost in USD, or 0.0 if not available
    """
    if not generation_id:
        return 0.0
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # Extract actual cost from generation data
                generation_data = data.get('data', {})
                actual_cost = generation_data.get('total_cost', 0.0)
                return float(actual_cost) if actual_cost else 0.0
            else:
                return 0.0
                
    except Exception as e:
        print(f"Error fetching actual cost for {generation_id}: {e}")
        return 0.0
