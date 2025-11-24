"""Unit tests for storage module."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from backend.storage import (
    ensure_data_dir,
    get_conversation_path,
    create_conversation,
    get_conversation,
    save_conversation,
    list_conversations,
    add_user_message,
    add_assistant_message,
    update_conversation_title,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    with patch("backend.storage.DATA_DIR", temp_dir):
        yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


def test_ensure_data_dir(temp_data_dir):
    """Test data directory creation."""
    # Remove the directory to test creation
    shutil.rmtree(temp_data_dir)
    assert not Path(temp_data_dir).exists()

    ensure_data_dir()

    assert Path(temp_data_dir).exists()


def test_get_conversation_path(temp_data_dir):
    """Test conversation file path generation."""
    path = get_conversation_path("test-id-123")

    assert path.endswith("test-id-123.json")
    assert temp_data_dir in path


def test_create_conversation(temp_data_dir):
    """Test creating a new conversation."""
    conversation = create_conversation("test-id-456")

    assert conversation["id"] == "test-id-456"
    assert conversation["title"] == "New Conversation"
    assert conversation["messages"] == []
    assert "created_at" in conversation

    # Verify file was created
    path = get_conversation_path("test-id-456")
    assert Path(path).exists()


def test_get_conversation_exists(temp_data_dir):
    """Test retrieving an existing conversation."""
    # Create a conversation first
    original = create_conversation("test-id-789")

    # Retrieve it
    conversation = get_conversation("test-id-789")

    assert conversation is not None
    assert conversation["id"] == "test-id-789"
    assert conversation["title"] == original["title"]


def test_get_conversation_not_exists(temp_data_dir):
    """Test retrieving a non-existent conversation returns None."""
    conversation = get_conversation("non-existent-id")

    assert conversation is None


def test_save_conversation(temp_data_dir):
    """Test saving a conversation."""
    conversation = {
        "id": "test-save",
        "title": "Test Title",
        "created_at": "2024-01-01T00:00:00",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    save_conversation(conversation)

    # Verify file exists and contains correct data
    path = get_conversation_path("test-save")
    assert Path(path).exists()

    with open(path, "r") as f:
        saved = json.load(f)

    assert saved["id"] == "test-save"
    assert saved["title"] == "Test Title"
    assert len(saved["messages"]) == 1


def test_list_conversations_empty(temp_data_dir):
    """Test listing conversations when none exist."""
    conversations = list_conversations()

    assert conversations == []


def test_list_conversations_multiple(temp_data_dir):
    """Test listing multiple conversations."""
    # Create several conversations
    create_conversation("conv-1")
    create_conversation("conv-2")
    create_conversation("conv-3")

    conversations = list_conversations()

    assert len(conversations) == 3
    assert all("id" in c for c in conversations)
    assert all("created_at" in c for c in conversations)
    assert all("title" in c for c in conversations)
    assert all("message_count" in c for c in conversations)


def test_list_conversations_sorted_by_date(temp_data_dir):
    """Test conversations are sorted by creation date (newest first)."""
    # Create conversations with different timestamps
    conv1 = create_conversation("conv-1")
    conv1["created_at"] = "2024-01-01T00:00:00"
    save_conversation(conv1)

    conv2 = create_conversation("conv-2")
    conv2["created_at"] = "2024-01-03T00:00:00"
    save_conversation(conv2)

    conv3 = create_conversation("conv-3")
    conv3["created_at"] = "2024-01-02T00:00:00"
    save_conversation(conv3)

    conversations = list_conversations()

    # Should be sorted newest first
    assert conversations[0]["id"] == "conv-2"
    assert conversations[1]["id"] == "conv-3"
    assert conversations[2]["id"] == "conv-1"


def test_list_conversations_message_count(temp_data_dir):
    """Test message count is correct in conversation list."""
    conv = create_conversation("conv-with-messages")
    conv["messages"] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "stage1": [], "stage2": [], "stage3": {}},
    ]
    save_conversation(conv)

    conversations = list_conversations()

    assert conversations[0]["message_count"] == 2


def test_add_user_message(temp_data_dir):
    """Test adding a user message to a conversation."""
    create_conversation("test-user-msg")

    add_user_message("test-user-msg", "What is AI?")

    conversation = get_conversation("test-user-msg")
    assert len(conversation["messages"]) == 1
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][0]["content"] == "What is AI?"


def test_add_user_message_nonexistent_conversation(temp_data_dir):
    """Test adding user message to non-existent conversation raises error."""
    with pytest.raises(ValueError, match="not found"):
        add_user_message("non-existent", "Hello")


def test_add_assistant_message(temp_data_dir):
    """Test adding an assistant message with all stages."""
    create_conversation("test-assistant-msg")

    stage1 = [{"model": "gpt-4", "response": "Test response"}]
    stage2 = [{"model": "gpt-4", "ranking": "FINAL RANKING:\n1. Response A"}]
    stage3 = {"model": "chairman", "response": "Final answer"}

    add_assistant_message("test-assistant-msg", stage1, stage2, stage3)

    conversation = get_conversation("test-assistant-msg")
    assert len(conversation["messages"]) == 1
    assert conversation["messages"][0]["role"] == "assistant"
    assert conversation["messages"][0]["stage1"] == stage1
    assert conversation["messages"][0]["stage2"] == stage2
    assert conversation["messages"][0]["stage3"] == stage3


def test_add_assistant_message_nonexistent_conversation(temp_data_dir):
    """Test adding assistant message to non-existent conversation raises error."""
    with pytest.raises(ValueError, match="not found"):
        add_assistant_message("non-existent", [], [], {})


def test_update_conversation_title(temp_data_dir):
    """Test updating conversation title."""
    create_conversation("test-title-update")

    update_conversation_title("test-title-update", "My Custom Title")

    conversation = get_conversation("test-title-update")
    assert conversation["title"] == "My Custom Title"


def test_update_conversation_title_nonexistent(temp_data_dir):
    """Test updating title of non-existent conversation raises error."""
    with pytest.raises(ValueError, match="not found"):
        update_conversation_title("non-existent", "Some Title")


def test_conversation_persistence(temp_data_dir):
    """Test that conversations persist across multiple operations."""
    # Create conversation
    create_conversation("persist-test")

    # Add user message
    add_user_message("persist-test", "First message")

    # Add assistant message
    add_assistant_message("persist-test", [], [], {})

    # Update title
    update_conversation_title("persist-test", "Persisted Conversation")

    # Add another user message
    add_user_message("persist-test", "Second message")

    # Verify all changes persisted
    conversation = get_conversation("persist-test")
    assert conversation["title"] == "Persisted Conversation"
    assert len(conversation["messages"]) == 3
    assert conversation["messages"][0]["content"] == "First message"
    assert conversation["messages"][2]["content"] == "Second message"


def test_list_conversations_ignores_non_json_files(temp_data_dir):
    """Test that list_conversations ignores non-JSON files."""
    # Create a conversation
    create_conversation("valid-conv")

    # Create a non-JSON file in the data directory
    non_json_path = Path(temp_data_dir) / "not_a_conversation.txt"
    non_json_path.write_text("This is not JSON")

    conversations = list_conversations()

    # Should only return the valid conversation
    assert len(conversations) == 1
    assert conversations[0]["id"] == "valid-conv"
