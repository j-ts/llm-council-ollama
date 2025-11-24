"""Job storage and management for background processing."""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

JOBS_DIR = "data/jobs"


def ensure_jobs_dir():
    """Ensure the jobs directory exists."""
    Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)


def get_job_path(job_id: str) -> str:
    """Get the file path for a job."""
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def create_job(job_id: str, conversation_id: str, user_query: str) -> Dict[str, Any]:
    """
    Create a new job.

    Args:
        job_id: Unique identifier for the job
        conversation_id: Associated conversation ID
        user_query: The user's question

    Returns:
        New job dict
    """
    ensure_jobs_dir()

    job = {
        "job_id": job_id,
        "conversation_id": conversation_id,
        "user_query": user_query,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None
    }

    save_job(job)
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a job from storage.

    Args:
        job_id: Unique identifier for the job

    Returns:
        Job dict or None if not found
    """
    path = get_job_path(job_id)

    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_job(job: Dict[str, Any]):
    """
    Save a job to storage.

    Args:
        job: Job dict to save
    """
    ensure_jobs_dir()

    path = get_job_path(job['job_id'])
    with open(path, 'w') as f:
        json.dump(job, f, indent=2)


def update_job_status(job_id: str, status: str, **kwargs):
    """
    Update a job's status and optional fields.

    Args:
        job_id: Job identifier
        status: New status (pending, processing, completed, failed)
        **kwargs: Additional fields to update
    """
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    job["status"] = status

    if status == "processing" and job.get("started_at") is None:
        job["started_at"] = datetime.utcnow().isoformat()
    elif status in ("completed", "failed"):
        job["completed_at"] = datetime.utcnow().isoformat()

    for key, value in kwargs.items():
        job[key] = value

    save_job(job)


def update_job_result(job_id: str, result: Dict[str, Any]):
    """
    Update a job with the result.

    Args:
        job_id: Job identifier
        result: Result data
    """
    update_job_status(job_id, "completed", result=result)


def update_job_error(job_id: str, error: str):
    """
    Update a job with an error.

    Args:
        job_id: Job identifier
        error: Error message
    """
    update_job_status(job_id, "failed", error=error)
