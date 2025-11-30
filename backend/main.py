"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

try:
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from rq import Queue
    from rq import Worker
except Exception:
    # Optional dependency: allow the API to run without Redis/RQ installed
    Redis = None
    Queue = None
    Worker = None
    RedisConnectionError = Exception
    print("Warning: redis/rq not installed; background jobs will run inline.")

from . import storage, jobs
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .config import REDIS_URL
from .worker import process_council_job

app = FastAPI(title="LLM Council API")

# Initialize Redis connection and RQ queue
redis_conn = None
task_queue = None

if Redis and Queue:
    try:
        redis_conn = Redis.from_url(REDIS_URL)
        task_queue = Queue("council", connection=redis_conn)
    except Exception as e:
        print(f"Warning: Unable to initialize Redis queue ({e}); falling back to inline jobs.")


def _has_active_worker(queue: Queue) -> bool:
    """
    Check if there is at least one worker listening on the given queue.

    Returns False if we cannot inspect workers (e.g., Redis down).
    """
    if not queue or not Worker:
        return False
    try:
        workers = Worker.all(queue=queue)
        return any(queue.name in worker.queue_names() for worker in workers)
    except Exception as e:
        print(f"Warning: Unable to inspect RQ workers ({e}); treating as no workers.")
        return False


async def _run_job_inline(job_id: str, conversation_id: str, user_query: str):
    """
    Run the council job in-process when Redis/RQ is unavailable.
    """
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, process_council_job, job_id, conversation_id, user_query
        )
    except Exception as e:
        # process_council_job already updates job status on failure
        print(f"Inline processing failed for job {job_id}: {e}")

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


def _validate_api_key_format(api_key_value: Any, provider_label: str):
    """Validate API key presence or env var reference format."""
    if isinstance(api_key_value, str) and api_key_value.startswith("env:"):
        env_var_name = api_key_value[4:].strip()
        if not env_var_name:
            raise HTTPException(status_code=400, detail=f"Environment variable name is required for {provider_label} API key")
        return
    if not api_key_value:
        raise HTTPException(status_code=400, detail=f"API key is required for {provider_label} models")


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

    # Enqueue the job for background processing, or fall back to inline execution
    enqueued = False
    if task_queue:
        try:
            if _has_active_worker(task_queue):
                task_queue.enqueue(
                    process_council_job,
                    job_id,
                    conversation_id,
                    request.content,
                    job_timeout='30m'  # 30 minutes timeout
                )
                enqueued = True
            else:
                print("No active RQ workers found; running job inline.")
        except (RedisConnectionError, Exception) as e:
            print(f"Failed to enqueue job {job_id}, running inline instead: {e}")

    if not enqueued:
        asyncio.create_task(_run_job_inline(job_id, conversation_id, request.content))

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
    return config_manager.get_config(mask_sensitive=True)


@app.post("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update configuration."""
    from .config import config_manager
    from .providers import ProviderFactory

    updated = config_manager.update_config(config)
    # Reset cached providers so credential changes take effect immediately
    ProviderFactory.clear_cache()
    return config_manager.get_config(mask_sensitive=True)


# Model Registry CRUD Endpoints

@app.get("/api/models")
async def get_models():
    """Get all configured models from the registry."""
    from .config import config_manager
    config = config_manager.get_config(mask_sensitive=True)
    return {"models": config.get("models", {})}


@app.post("/api/models")
async def add_model(model_data: Dict[str, Any]):
    """Add a new model to the registry."""
    from .config import config_manager
    from .providers import ProviderFactory
    import uuid
    
    config = config_manager.get_config()
    models = config.get("models", {})
    ollama_settings = config.get("ollama_settings", {})
    
    # Generate ID
    model_id = str(uuid.uuid4())
    
    # Validate required fields based on type
    model_type = model_data.get("type")
    if not model_type:
        raise HTTPException(status_code=400, detail="Model type is required")
    
    if model_type not in ["ollama", "openrouter", "openai-compatible"]:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
    
    if not model_data.get("label"):
        raise HTTPException(status_code=400, detail="Label is required")
    
    if not model_data.get("model_name"):
        raise HTTPException(status_code=400, detail="Model name is required")
    
    # Type-specific validation
    api_key_value = model_data.get("api_key", "")

    if model_type == "ollama":
        if not model_data.get("base_url"):
            raise HTTPException(status_code=400, detail="Base URL is required for Ollama models")
    elif model_type == "openrouter":
        _validate_api_key_format(api_key_value, "OpenRouter")
    elif model_type == "openai-compatible":
        if not model_data.get("base_url"):
            raise HTTPException(status_code=400, detail="Base URL is required for OpenAI-compatible models")
        _validate_api_key_format(api_key_value, "OpenAI-compatible")
    
    # Add model
    models[model_id] = {
        "label": model_data.get("label"),
        "type": model_type,
        "model_name": model_data.get("model_name"),
        "base_url": model_data.get("base_url"),
        "api_key": model_data.get("api_key")
    }

    # Validate provider creation early to surface env var errors
    try:
        ProviderFactory.get_provider_from_model_config(models[model_id], ollama_settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    config["models"] = models
    config_manager.update_config(config)
    
    # Clear provider cache
    ProviderFactory.clear_cache()
    
    sanitized_config = config_manager.get_config(mask_sensitive=True)
    return {"model_id": model_id, "model": sanitized_config.get("models", {}).get(model_id, {})}


@app.put("/api/models/{model_id}")
async def update_model(model_id: str, model_data: Dict[str, Any]):
    """Update an existing model in the registry."""
    from .config import config_manager
    from .providers import ProviderFactory
    
    config = config_manager.get_config()
    models = config.get("models", {})
    ollama_settings = config.get("ollama_settings", {})
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    existing_model = models.get(model_id, {})
    proposed_model = {**existing_model, **model_data}

    if "api_key" in model_data:
        incoming_key = model_data.get("api_key", "")
        if incoming_key == "***":
            proposed_model["api_key"] = existing_model.get("api_key", "")
        else:
            proposed_model["api_key"] = incoming_key

    model_type = proposed_model.get("type", existing_model.get("type"))
    api_key_value = proposed_model.get("api_key", "")

    if model_type == "ollama":
        if not proposed_model.get("base_url"):
            raise HTTPException(status_code=400, detail="Base URL is required for Ollama models")
    elif model_type == "openrouter":
        _validate_api_key_format(api_key_value, "OpenRouter")
    elif model_type == "openai-compatible":
        if not proposed_model.get("base_url"):
            raise HTTPException(status_code=400, detail="Base URL is required for OpenAI-compatible models")
        _validate_api_key_format(api_key_value, "OpenAI-compatible")

    # Validate provider creation to surface env var errors early
    try:
        ProviderFactory.get_provider_from_model_config(proposed_model, ollama_settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    models[model_id] = proposed_model
    config["models"] = models
    config_manager.update_config(config)
    
    # Clear provider cache
    ProviderFactory.clear_cache()
    
    sanitized_config = config_manager.get_config(mask_sensitive=True)
    return {"model_id": model_id, "model": sanitized_config.get("models", {}).get(model_id, {})}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    from .config import config_manager
    
    config = config_manager.get_config()
    models = config.get("models", {})
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if model is in use
    council_models = config.get("council_models", [])
    chairman_model = config.get("chairman_model")
    
    if model_id in council_models:
        raise HTTPException(status_code=400, detail="Model is in use by council")
    if model_id == chairman_model:
        raise HTTPException(status_code=400, detail="Model is in use as chairman")
    
    # Delete model
    del models[model_id]
    
    config["models"] = models
    config_manager.update_config(config)
    
    # Clear provider cache
    from .providers import ProviderFactory
    ProviderFactory.clear_cache()
    
    return {"message": "Model deleted successfully"}


@app.get("/api/models/all")
async def list_all_models():
    """List available models from all configured providers."""
    from .config import config_manager
    from .providers import ProviderFactory
    
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    all_models = {}

    try:
        # Get all configured providers
        providers = ProviderFactory.get_all_providers(providers_config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
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
async def list_models(provider: str, config_name: str = None):
    """List available models for a specific provider."""
    from .config import config_manager
    from .providers import ProviderFactory
    
    config = config_manager.get_config()
    providers_config = config.get("providers", {})
    
    if provider not in providers_config:
        return {"models": []}
    
    # Create a dummy model config to get the provider instance
    dummy_model_config = {"name": "dummy", "provider": provider}
    if config_name:
        dummy_model_config["openai_config_name"] = config_name
    
    try:
        provider_instance = ProviderFactory.get_provider_for_model(dummy_model_config, providers_config)
        models = await provider_instance.list_models()
        return {"models": models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error listing models for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
