"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
import asyncio
from .config import config_manager
from .providers import ProviderFactory, ModelProvider

async def query_models_parallel(
    model_configs: List[Dict[str, Any]],
    providers_config: Dict[str, Any],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel, each with their own provider.
    """
    async def query_safe(model_config):
        try:
            provider = ProviderFactory.get_provider_for_model(model_config, providers_config)
            return await provider.query(model_config["name"], messages)
        except Exception as e:
            print(f"Error querying model {model_config.get('name')}: {e}")
            return None

    tasks = [query_safe(model_config) for model_config in model_configs]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Map responses by model name
    result = {}
    for model_config, response in zip(model_configs, responses):
        if isinstance(response, Exception):
            print(f"Exception for model {model_config.get('name')}: {response}")
            result[model_config["name"]] = None
        else:
            result[model_config["name"]] = response
    
    return result

async def stage1_collect_responses(user_query: str) -> Tuple[List[Dict[str, Any]], float, Dict[str, str]]:
    """
    Stage 1: Collect individual responses from all council models.
    """
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    council_models = config.get("council_models", [])

    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel (each with their own provider)
    responses = await query_models_parallel(council_models, providers_config, messages)

    # Format results and collect costs + generation IDs
    stage1_results = []
    total_cost = 0.0
    generation_ids = {}
    
    for model_name, response in responses.items():
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
                    generation_ids[f"stage1_{model_name}"] = gen_id
            
            stage1_results.append({
                "model": model_name,
                "response": response.get('content', ''),
                "cost": model_cost,
                "cost_status": cost_status
            })

    return stage1_results, total_cost, generation_ids


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], float, Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.
    """
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    council_models = config.get("council_models", [])

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
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
    responses = await query_models_parallel(council_models, providers_config, messages)

    # Format results and collect costs + generation IDs
    stage2_results = []
    total_cost = 0.0
    generation_ids = {}
    
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            
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
                    generation_ids[f"stage2_{model}"] = gen_id
            
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed,
                "cost": model_cost,
                "cost_status": cost_status
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
    providers_config = config.get("providers", {})
    chairman_model_config = config.get("chairman_model", {})
    chairman_model_name = chairman_model_config.get("name") if isinstance(chairman_model_config, dict) else chairman_model_config

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with its specific provider
    try:
        provider = ProviderFactory.get_provider_for_model(chairman_model_config, providers_config)
        response = await provider.query(chairman_model_name, messages)
    except Exception as e:
        print(f"Error querying chairman model: {e}")
        response = None

    if response is None:
        # Fallback if chairman fails
        return {
            "model": chairman_model_name,
            "response": "Error: Unable to generate final synthesis."
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
        generation_ids[f"stage3_{chairman_model_name}"] = gen_id

    return {
        "model": chairman_model_name,
        "response": response.get('content', ''),
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
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.
    """
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    
    # Use the chairman model for title generation
    chairman_model_config = config.get("chairman_model", {})
    
    # Handle both old and new formats
    if isinstance(chairman_model_config, dict):
        title_model_config = chairman_model_config
        title_model_name = chairman_model_config.get("name", "")
    else:
        # Old format: chairman_model is a string
        title_model_name = chairman_model_config
        # Assume openrouter provider for legacy configs
        title_model_config = {"name": title_model_name, "provider": "openrouter"}
    
    if not title_model_name:
        return "New Conversation"
        
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    try:
        provider = ProviderFactory.get_provider_for_model(title_model_config, providers_config)
        response = await provider.query(title_model_name, messages, timeout=30.0)
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
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict, float]:
    """
    Run the complete 3-stage council process.
    """
    total_cost = 0.0
    
    # Stage 1: Collect individual responses
    stage1_results, stage1_cost, stage1_gen_ids = await stage1_collect_responses(user_query)
    total_cost += stage1_cost

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}, 0.0

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
