"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio
from redis import Redis
from rq import Queue

from . import storage, jobs
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .config import REDIS_URL
from .worker import process_council_job

app = FastAPI(title="LLM Council API")

# Initialize Redis connection and RQ queue
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue("council", connection=redis_conn)

# Enable CORS for local and network development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int
    total_cost: float


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    storage.delete_conversation(conversation_id)
    return {"message": "Conversation deleted successfully"}


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata, total_cost = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )
    
    # Update conversation cost
    storage.add_cost_to_conversation(conversation_id, total_cost)

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata,
        "cost": total_cost
    }


@app.post("/api/conversations/{conversation_id}/message/async")
async def send_message_async(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and process it in the background.
    Returns immediately with a job_id that can be used to check status.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create job record
    jobs.create_job(job_id, conversation_id, request.content)

    # Enqueue the job for background processing
    task_queue.enqueue(
        process_council_job,
        job_id,
        conversation_id,
        request.content,
        job_timeout='30m'  # 30 minutes timeout
    )

    # Start title generation in parallel if first message
    if is_first_message:
        asyncio.create_task(generate_and_update_title(conversation_id, request.content))

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued for processing"
    }


async def generate_and_update_title(conversation_id: str, user_query: str):
    """Helper function to generate and update conversation title."""
    title = await generate_conversation_title(user_query)
    storage.update_conversation_title(conversation_id, title)


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and result of a background job.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Track total cost
            total_cost = 0.0
            
            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage1_cost = await stage1_collect_responses(request.content)
            total_cost += stage1_cost
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model, stage2_cost = await stage2_collect_rankings(request.content, stage1_results)
            total_cost += stage2_cost
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result, stage3_cost = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
            total_cost += stage3_cost
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )
            
            # Update conversation cost
            storage.add_cost_to_conversation(conversation_id, total_cost)

            # Send completion event with cost
            yield f"data: {json.dumps({'type': 'complete', 'cost': total_cost})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    from .config import config_manager
    return config_manager.get_config()


@app.post("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update configuration."""
    from .config import config_manager
    return config_manager.update_config(config)


@app.get("/api/models/all")
async def list_all_models():
    """List available models from all configured providers."""
    from .config import config_manager
    from .providers import ProviderFactory
    
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    
    all_models = {}
    
    # Get all configured providers
    providers = ProviderFactory.get_all_providers(providers_config)
    
    # Query each provider for its models
    for provider_name, provider_instance in providers.items():
        try:
            models = await provider_instance.list_models()
            all_models[provider_name] = models
        except Exception as e:
            print(f"Error listing models for {provider_name}: {e}")
            all_models[provider_name] = []
    
    return {"models": all_models}


@app.get("/api/models/{provider}")
async def list_models(provider: str):
    """List available models for a specific provider."""
    from .config import config_manager
    from .providers import ProviderFactory
    
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    
    if provider not in providers_config:
        return {"models": []}
    
    # Create a dummy model config to get the provider instance
    dummy_model_config = {"name": "dummy", "provider": provider}
    
    try:
        provider_instance = ProviderFactory.get_provider_for_model(dummy_model_config, providers_config)
        models = await provider_instance.list_models()
        return {"models": models}
    except Exception as e:
        print(f"Error listing models for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

