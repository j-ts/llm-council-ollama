"""Background job to update estimated costs with actual costs from OpenRouter."""

import asyncio
import json
from typing import Dict, Any
from pathlib import Path

from .openrouter import fetch_actual_cost
from .storage import get_conversation, save_conversation
from .config import DATA_DIR


async def update_conversation_costs(conversation_id: str, generation_ids: Dict[str, str]):
    """
    Poll OpenRouter generation API and update conversation with actual costs.
    
    Args:
        conversation_id: ID of the conversation to update
        generation_ids: Dict mapping stage+model to generation_id
    """
    # Wait a bit for OpenRouter to calculate costs
    await asyncio.sleep(3)
    
    # Fetch all actual costs
    actual_costs = {}
    for key, gen_id in generation_ids.items():
        if gen_id:
            actual_cost = await fetch_actual_cost(gen_id)
            if actual_cost > 0:
                actual_costs[key] = actual_cost
    
    if not actual_costs:
        # No actual costs available yet, skip update
        return
    
    # Load conversation
    conversation = get_conversation(conversation_id)
    if not conversation:
        return
    
    # Update costs in the last assistant message
    messages = conversation.get('messages', [])
    if not messages or messages[-1]['role'] != 'assistant':
        return
    
    assistant_msg = messages[-1]
    total_updated_cost = 0.0
    
    # Update stage1 costs
    if 'stage1' in assistant_msg:
        for i, resp in enumerate(assistant_msg['stage1']):
            key = f"stage1_{resp['model']}"
            if key in actual_costs:
                resp['cost'] = actual_costs[key]
                resp['cost_status'] = 'actual'
                total_updated_cost += actual_costs[key]
    
    # Update stage2 costs
    if 'stage2' in assistant_msg:
        for i, rank in enumerate(assistant_msg['stage2']):
            key = f"stage2_{rank['model']}"
            if key in actual_costs:
                rank['cost'] = actual_costs[key]
                rank['cost_status'] = 'actual'
                total_updated_cost += actual_costs[key]
    
    # Update stage3 cost
    if 'stage3' in assistant_msg:
        key = f"stage3_{assistant_msg['stage3']['model']}"
        if key in actual_costs:
            assistant_msg['stage3']['cost'] = actual_costs[key]
            assistant_msg['stage3']['cost_status'] = 'actual'
            total_updated_cost += actual_costs[key]
    
    # Update stage_costs in metadata
    if 'metadata' in assistant_msg and 'stage_costs' in assistant_msg['metadata']:
        stage_costs = assistant_msg['metadata']['stage_costs']
        stage_costs['status'] = 'actual'
        
        # Recalculate stage totals from actual costs
        stage1_total = sum(r.get('cost', 0) for r in assistant_msg.get('stage1', []))
        stage2_total = sum(r.get('cost', 0) for r in assistant_msg.get('stage2', []))
        stage3_cost = assistant_msg.get('stage3', {}).get('cost', 0)
        
        stage_costs['stage1'] = stage1_total
        stage_costs['stage2'] = stage2_total
        stage_costs['stage3'] = stage3_cost
        stage_costs['total'] = stage1_total + stage2_total + stage3_cost
    
    # Update conversation total_cost
    if total_updated_cost > 0:
        conversation['total_cost'] = total_updated_cost
    
    # Save updated conversation
    save_conversation(conversation)
    print(f"✅ Updated conversation {conversation_id} with actual costs: ${total_updated_cost:.5f}")


def schedule_cost_update(conversation_id: str, generation_ids: Dict[str, str]):
    """
    Schedule a background task to update costs.
    This is called after a conversation is created with estimated costs.
    
    Args:
        conversation_id: ID of the conversation
        generation_ids: Dict mapping stage+model to generation_id
    """
    # Create async task (will run in background)
    asyncio.create_task(update_conversation_costs(conversation_id, generation_ids))
