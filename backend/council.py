"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
import asyncio
from .config import config_manager
from .providers import ProviderFactory


def resolve_model_configs_from_ids(model_ids: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Resolve model IDs to full model configurations from registry.
    
    Args:
        model_ids: List of model IDs (may contain old format dicts or new format string IDs)
        config: Full configuration with models registry
    
    Returns:
        List of model configs with registry data
    """
    models_registry = config.get("models", {})
    resolved = []
    
    for model_id in model_ids:
        # Skip old format (dicts) - only process new format (string IDs)
        if not isinstance(model_id, str):
            print(f"Warning: Skipping old format model config in council_models: {model_id}")
            continue
            
        if model_id in models_registry:
            model_config = models_registry[model_id].copy()
            model_config["id"] = model_id
            resolved.append(model_config)
    
    return resolved


def build_model_key(model_config: Dict[str, Any]) -> str:
    """
    Build a stable identifier for a model configuration.
    Includes provider and OpenAI config name (when present) to avoid collisions.
    """
    # Support both new registry format (type/model_name) and legacy (provider/name)
    provider = model_config.get("type") or model_config.get("provider", "openrouter")
    model_name = model_config.get("model_name") or model_config.get("name", "")
    parts = [provider]

    if provider in ("openai", "openai-compatible"):
        parts.append(model_config.get("openai_config_name", "default"))

    parts.append(model_name)
    return "::".join(parts)


def build_model_display_name(model_config: Dict[str, Any]) -> str:
    """
    Human-friendly label that includes the provider/config when useful.
    """
    # Support both new registry format (type/model_name) and legacy (provider/name)
    model_name = model_config.get("model_name") or model_config.get("name", "")
    provider = model_config.get("type") or model_config.get("provider", "openrouter")

    if provider in ("openai", "openai-compatible"):
        config_name = model_config.get("openai_config_name") or "OpenAI"
        return f"{model_name} ({config_name})"
    if provider == "openrouter":
        return f"{model_name} (OpenRouter)"
    if provider == "ollama":
        return f"{model_name} (Ollama)"
    return f"{model_name} ({provider})"

def sanitize_model_options(model_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Normalize the options dict for a model and validate known keys.

    Returns a (options, error) tuple. The options dict is safe to pass to providers.
    """
    if not isinstance(model_config, dict):
        return {}, None

    raw_options = model_config.get("options")
    if raw_options in (None, ""):
        return {}, None
    if not isinstance(raw_options, dict):
        return {}, "Options must be an object/dictionary"

    options = {k: v for k, v in raw_options.items() if v is not None}

    if "num_ctx" in options:
        try:
            num_ctx_value = int(options["num_ctx"])
        except (TypeError, ValueError):
            return {}, "num_ctx must be a positive integer"
        if num_ctx_value <= 0:
            return {}, "num_ctx must be a positive integer"
        options["num_ctx"] = num_ctx_value

    return options, None


async def query_models_parallel(
    model_configs: List[Dict[str, Any]],
    ollama_settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    serialize_ollama: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models from the registry, each with their own provider.
    
    Args:
        model_configs: List of model configs from registry (with id, type, model_name, etc.)
        ollama_settings: Global Ollama settings
        messages: Chat messages
        serialize_ollama: Whether to serialize Ollama requests
    """
    async def query_safe(model_config: Dict[str, Any]):
        model_id = model_config.get("id", "unknown")
        model_name = model_config.get("model_name", "")
        model_label = model_config.get("label", model_name)
        
        try:
            provider = ProviderFactory.get_provider_from_model_config(model_config, ollama_settings)
            response = await provider.query(model_name, messages)
            return {"response": response, "error": None, "model_config": model_config}
        except Exception as e:
            error_message = str(e)
            print(f"Error querying model {model_label}: {error_message}")
            return {"response": None, "error": error_message, "model_config": model_config}

    result = {}

    def _store(model_config, response):
        model_id = model_config.get("id", "unknown")
        result[model_id] = {
            "response": response.get("response") if isinstance(response, dict) else None,
            "error": response.get("error") if isinstance(response, dict) else "Unknown error",
            "model_config": model_config
        }

    if serialize_ollama:
        ollama_models = [mc for mc in model_configs if mc.get("type") == "ollama"]
        other_models = [mc for mc in model_configs if mc.get("type") != "ollama"]

        if other_models:
            other_tasks = [query_safe(model_config) for model_config in other_models]
            other_responses = await asyncio.gather(*other_tasks)
            for model_config, response in zip(other_models, other_responses):
                _store(model_config, response)

        # Run local/Ollama models one at a time to limit simultaneous GPU pressure
        for model_config in ollama_models:
            response = await query_safe(model_config)
            _store(model_config, response)
    else:
        tasks = [query_safe(model_config) for model_config in model_configs]
        responses = await asyncio.gather(*tasks)
        for model_config, response in zip(model_configs, responses):
            _store(model_config, response)
    
    return result

async def stage1_collect_responses(user_query: str) -> Tuple[List[Dict[str, Any]], float, Dict[str, str]]:
    """
    Stage 1: Collect individual responses from all council models.
    """
    config = config_manager.get_config()
    
    # Resolve model IDs to full configs
    council_model_ids = config.get("council_models", [])
    council_models = resolve_model_configs_from_ids(council_model_ids, config)
    
    ollama_settings = config.get("ollama_settings", {})
    serialize_ollama = ollama_settings.get("serialize_requests", False)

    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel (each with their own provider)
    responses = await query_models_parallel(
        council_models,
        ollama_settings,
        messages,
        serialize_ollama=serialize_ollama
    )

    # Format results and collect costs + generation IDs
    stage1_results = []
    total_cost = 0.0
    generation_ids = {}
    
    for model_id, data in responses.items():
        response = data.get("response")
        error = data.get("error")
        model_config = data.get("model_config", {})
        model_label = model_config.get("label", model_id)

        if response is not None:
            # Extract cost and generation ID
            model_cost = 0.0
            cost_status = 'estimated'
            gen_id = None
            
            if 'usage' in response:
                usage = response['usage']
                model_cost = usage.get('cost', 0.0)
                cost_status = usage.get('cost_status', 'estimated')
                gen_id = usage.get('generation_id')
                total_cost += model_cost
                
                if gen_id:
                    generation_ids[f"stage1_{model_id}"] = gen_id
            
            stage1_results.append({
                "model_id": model_id,
                "model": model_config.get("model_name", model_id),
                "model_display": model_label,
                "response": response.get('content', ''),
                "cost": model_cost,
                "cost_status": cost_status,
                "error": None
            })
        else:
            stage1_results.append({
                "model_id": model_id,
                "model": model_config.get("model_name", model_id),
                "model_display": model_label,
                "response": "",
                "cost": 0.0,
                "cost_status": "error",
                "error": error or "Failed to query model"
            })

    return stage1_results, total_cost, generation_ids


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], float, Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.
    """
    config = config_manager.get_config()
    
    # Resolve model IDs to full configs
    council_model_ids = config.get("council_models", [])
    council_models = resolve_model_configs_from_ids(council_model_ids, config)
    
    ollama_settings = config.get("ollama_settings", {})
    serialize_ollama = ollama_settings.get("serialize_requests", False)

    valid_stage1_results = [result for result in stage1_results if not result.get("error")]
    if not valid_stage1_results:
        return [], {}, 0.0, {}

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(valid_stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": {
            "id": result.get("model_id", result.get("model")),
            "model": result.get("model"),
            "model_display": result.get("model_display") or result.get("model"),
            "provider": result.get("provider"),
            "openai_config_name": result.get("openai_config_name"),
        }
        for label, result in zip(labels, valid_stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, valid_stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Do not preface with acknowledgments or chit-chat. Start immediately with the evaluations, then the FINAL RANKING block exactly as specified.

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(
        council_models,
        ollama_settings,
        messages,
        serialize_ollama=serialize_ollama
    )

    # Format results and collect costs + generation IDs
    stage2_results = []
    total_cost = 0.0
    generation_ids = {}
    
    for model_key, data in responses.items():
        response = data.get("response")
        error = data.get("error")
        model_config = data.get("model_config", {})

        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            parsed_error = None
            if not parsed:
                parsed_error = "Ranking not found in model output"
            
            # Extract cost and generation ID
            model_cost = 0.0
            cost_status = 'estimated'
            gen_id = None
            
            if 'usage' in response:
                usage = response['usage']
                model_cost = usage.get('cost', 0.0)
                cost_status = usage.get('cost_status', 'estimated')
                gen_id = usage.get('generation_id')
                total_cost += model_cost
                
                if gen_id:
                    generation_ids[f"stage2_{model_key}"] = gen_id
            
            stage2_results.append({
                "model_id": model_key,
                "model": model_config.get("name", model_key),
                "model_display": build_model_display_name(model_config),
                "provider": model_config.get("provider"),
                "openai_config_name": model_config.get("openai_config_name"),
                "ranking": full_text,
                "parsed_ranking": parsed,
                "cost": model_cost,
                "cost_status": cost_status,
                "error": parsed_error
            })
        else:
            stage2_results.append({
                "model_id": model_key,
                "model": model_config.get("name", model_key),
                "model_display": build_model_display_name(model_config),
                "provider": model_config.get("provider"),
                "openai_config_name": model_config.get("openai_config_name"),
                "ranking": "",
                "parsed_ranking": [],
                "cost": 0.0,
                "cost_status": "error",
                "error": error or "Failed to query model"
            })

    return stage2_results, label_to_model, total_cost, generation_ids


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float, Dict[str, str]]:
    """
    Stage 3: Chairman synthesizes final response.
    """
    config = config_manager.get_config()
    ollama_settings = config.get("ollama_settings", {})
    
    # Resolve chairman model ID
    chairman_model_id = config.get("chairman_model")
    if not chairman_model_id:
        return {
            "model_id": "error",
            "model": "error",
            "model_display": "No chairman configured",
            "response": "No chairman model configured. Please add a model and set it as chairman.",
            "error": "No chairman model configured"
        }, 0.0, {}
    
    models_registry = config.get("models", {})
    chairman_config = models_registry.get(chairman_model_id)
    if not chairman_config:
        return {
            "model_id": "error",
            "model": "error",
            "model_display": "Chairman not found",
            "response": f"Chairman model {chairman_model_id} not found in registry.",
            "error": "Chairman model not found"
        }, 0.0, {}
    
    chairman_config = chairman_config.copy()
    chairman_config["id"] = chairman_model_id
    chairman_model_name = chairman_config.get("model_name", chairman_model_id)
    chairman_model_label = chairman_config.get("label", chairman_model_name)

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result.get('model_display', result.get('model'))}\nResponse: {result['response']}"
        for result in stage1_results
        if not result.get("error")
    ])

    stage2_text = "\n\n".join([
        f"Model: {result.get('model_display', result.get('model'))}\nRanking: {result['ranking']}"
        for result in stage2_results
        if not result.get("error")
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses. Your job is to deliver the final response to the user. 

Original Question: {user_query}

STAGE 1 - Individual Responses (treat these as raw ideas, do NOT cite "Response A/B/etc."):
{stage1_text}

STAGE 2 - Peer Rankings (use these only to gauge confidence/quality, not as content to parrot):
{stage2_text}

Your response must:
1. Directly answer the user's original question in plain language. Never refer to "Response A/B/C" or similar labels.
2. Integrate the strongest points from the Stage 1 answers, resolving disagreements if they exist. Explain the reasoning so the user understands the conclusion without seeing the earlier responses.
3. Explicitly mention supporting evidence or caveats the council surfaced, but use natural prose (e.g., "Some models noted...", "One perspective emphasized...").
4. Conclude with a concise takeaway or recommendation when appropriate.

Provide a single, comprehensive, accurate answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with its specific provider
    error_message = None
    response = None
    try:
        provider = ProviderFactory.get_provider_from_model_config(chairman_config, ollama_settings)
        response = await provider.query(chairman_model_name, messages)
    except Exception as e:
        print(f"Error querying chairman model: {e}")
        response = None
        error_message = str(e)

    if response is None:
        fallback_message = "Error: Unable to generate final synthesis."
        if error_message:
            fallback_message += f" ({error_message})"
        # Fallback if chairman fails
        return {
            "model_id": chairman_model_id,
            "model": chairman_model_name,
            "model_display": chairman_model_label,
            "response": fallback_message,
            "error": error_message or "Failed to generate synthesis",
            "cost": 0.0,
            "cost_status": "error"
        }, 0.0, {}

    # Extract cost and generation ID
    cost = 0.0
    cost_status = 'estimated'
    gen_id = None
    
    if response and 'usage' in response:
        usage = response['usage']
        cost = usage.get('cost', 0.0)
        cost_status = usage.get('cost_status', 'estimated')
        gen_id = usage.get('generation_id')
    
    generation_ids = {}
    if gen_id:
        generation_ids[f"stage3_{chairman_model_id}"] = gen_id

    return {
        "model_id": chairman_model_id,
        "model": chairman_model_name,
        "model_display": chairman_model_label,
        "response": response.get('content', ''),
        "error": None,
        "cost": cost,
        "cost_status": cost_status
    }, cost, generation_ids


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.
    """
    from collections import defaultdict

    def _normalize_label(entry: Any) -> Tuple[str, str, str]:
        """Return (id, base_model, display_name) for a label entry."""
        if isinstance(entry, dict):
            # Helper to safely get string value
            def get_str(key, default=None):
                val = entry.get(key)
                if isinstance(val, (str, int, float)):
                    return str(val)
                return default

            model_id = get_str("id") or get_str("model_id") or get_str("model")
            base_model = get_str("model") or model_id
            display = get_str("model_display") or base_model or model_id
            
            # Fallback if everything is missing or invalid
            if not model_id:
                model_id = "unknown_model"
            if not base_model:
                base_model = "unknown_model"
            if not display:
                display = "Unknown Model"
                
            return model_id, base_model, display
        
        # If entry is not a dict, convert to string
        entry_str = str(entry)
        return entry_str, entry_str, entry_str

    # Track positions for each model (by id) and store display names
    model_positions = defaultdict(list)
    model_display_lookup: Dict[str, Tuple[str, str]] = {}

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_id, base_model, display = _normalize_label(label_to_model[label])
                model_positions[model_id].append(position)
                model_display_lookup[model_id] = (base_model, display)

    # Calculate average position for each model
    aggregate = []
    for model_id, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            base_model, display_name = model_display_lookup.get(model_id, (model_id, model_id))
            aggregate.append({
                "model_id": model_id,
                "model": display_name,
                "model_display": display_name,
                "base_model": base_model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for the conversation using the chairman model.
    """
    config = config_manager.get_config()
    ollama_settings = config.get("ollama_settings", {})
    
    # Resolve chairman model ID
    chairman_model_id = config.get("chairman_model")
    if not chairman_model_id:
        return "New Conversation"
    
    models_registry = config.get("models", {})
    chairman_config = models_registry.get(chairman_model_id)
    if not chairman_config:
        return "New Conversation"
    
    chairman_config = chairman_config.copy()
    chairman_config["id"] = chairman_model_id
    chairman_model_name = chairman_config.get("model_name", chairman_model_id)
        
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    try:
        provider = ProviderFactory.get_provider_from_model_config(chairman_config, ollama_settings)
        response = await provider.query(chairman_model_name, messages, timeout=30.0)
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Conversation"

    if response is None:
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + '...'

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict, float]:
    """
    Run the complete 3-stage council process.
    """
    total_cost = 0.0
    
    # Stage 1: Collect individual responses
    stage1_results, stage1_cost, stage1_gen_ids = await stage1_collect_responses(user_query)
    total_cost += stage1_cost

    any_success = any(not result.get("error") for result in stage1_results)
    if not any_success:
        stage2_results: List[Dict[str, Any]] = []
        stage3_result = {
            "model_id": "error",
            "model": "error",
            "model_display": "error",
            "response": "All council models failed to respond. Check your provider settings or pull the missing Ollama models.",
            "error": "All models failed to respond.",
            "cost": 0.0,
            "cost_status": "error"
        }
        metadata = {
            "label_to_model": {},
            "aggregate_rankings": [],
            "stage_costs": {
                "stage1": stage1_cost,
                "stage2": 0.0,
                "stage3": 0.0,
                "total": total_cost,
                "status": "estimated"
            },
            "generation_ids": {**stage1_gen_ids}
        }
        return stage1_results, stage2_results, stage3_result, metadata, total_cost

    # Stage 2: Collect rankings
    stage2_results, label_to_model, stage2_cost, stage2_gen_ids = await stage2_collect_rankings(user_query, stage1_results)
    total_cost += stage2_cost

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result, stage3_cost, stage3_gen_ids = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    total_cost += stage3_cost
    
    # Merge all generation IDs
    all_generation_ids = {**stage1_gen_ids, **stage2_gen_ids, **stage3_gen_ids}
    
    # Prepare metadata with stage costs and generation IDs
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "stage_costs": {
            "stage1": stage1_cost,
            "stage2": stage2_cost,
            "stage3": stage3_cost,
            "total": total_cost,
            "status": "estimated"
        },
        "generation_ids": all_generation_ids
    }

    return stage1_results, stage2_results, stage3_result, metadata, total_cost
