"""
Tests for Memory Store - Save, load, and search functionality.

Tests memory persistence, retrieval, and search operations.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory import MemoryStore


def test_memory_store_creation():
    """Test that MemoryStore can be created."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        assert store.storage_path == Path(temp_path)
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_save_and_load():
    """Test saving and loading data from memory store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Test data
        test_data = {
            "user_id": "test_user",
            "name": "Test User",
            "level": "intermediate",
            "preferences": {
                "learning_style": "visual",
                "difficulty": "intermediate"
            },
            "topics_studied": ["linear_regression", "neural_networks"]
        }
        
        # Save
        store.save("test_key", test_data)
        
        # Load
        loaded_data = store.load("test_key")
        
        # Assert equality
        assert loaded_data == test_data
        assert loaded_data["user_id"] == "test_user"
        assert loaded_data["name"] == "Test User"
        assert loaded_data["level"] == "intermediate"
        assert loaded_data["preferences"]["learning_style"] == "visual"
        assert len(loaded_data["topics_studied"]) == 2
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_load_nonexistent_key():
    """Test loading a key that doesn't exist returns default."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Load non-existent key with default
        result = store.load("nonexistent_key", default="default_value")
        assert result == "default_value"
        
        # Load non-existent key without default
        result = store.load("nonexistent_key")
        assert result is None
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_append_to_list():
    """Test appending items to a list in memory store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Append first item (creates list)
        store.append_to_list("test_list", "item1")
        result = store.load("test_list")
        assert result == ["item1"]
        
        # Append second item
        store.append_to_list("test_list", "item2")
        result = store.load("test_list")
        assert result == ["item1", "item2"]
        
        # Append third item
        store.append_to_list("test_list", "item3")
        result = store.load("test_list")
        assert result == ["item1", "item2", "item3"]
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_append_to_existing_list():
    """Test appending to an existing list."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Save initial list
        initial_list = ["existing1", "existing2"]
        store.save("test_list", initial_list)
        
        # Append to existing list
        store.append_to_list("test_list", "new_item")
        result = store.load("test_list")
        
        assert result == ["existing1", "existing2", "new_item"]
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_search_by_tag():
    """Test searching entries by tag."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Save entries with tags
        store.save("entry1", {
            "data": "test1",
            "tags": ["tag1", "tag2"]
        })
        
        store.save("entry2", {
            "data": "test2",
            "tags": ["tag2", "tag3"]
        })
        
        store.save("entry3", {
            "data": "test3",
            "tags": ["tag1", "tag3"]
        })
        
        # Search by tag
        results = store.search_by_tag("tag1")
        assert len(results) == 2  # entry1 and entry3
        
        results = store.search_by_tag("tag2")
        assert len(results) == 2  # entry1 and entry2
        
        results = store.search_by_tag("tag3")
        assert len(results) == 2  # entry2 and entry3
        
        # Verify result structure
        for result in results:
            assert "key" in result
            assert "value" in result
            assert "tags" in result["value"]
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_overwrite_existing_key():
    """Test that saving to existing key overwrites the value."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = MemoryStore(storage_path=temp_path, use_tinydb=False)
        
        # Save initial value
        store.save("test_key", {"value": "original"})
        assert store.load("test_key")["value"] == "original"
        
        # Overwrite with new value
        store.save("test_key", {"value": "updated"})
        assert store.load("test_key")["value"] == "updated"
        
        store.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_store_persistence():
    """Test that data persists across store instances."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        # Create first store instance and save data
        store1 = MemoryStore(storage_path=temp_path, use_tinydb=False)
        test_data = {"persistent": "data", "number": 42}
        store1.save("persistent_key", test_data)
        store1.close()
        
        # Create second store instance and load data
        store2 = MemoryStore(storage_path=temp_path, use_tinydb=False)
        loaded_data = store2.load("persistent_key")
        
        # Assert equality
        assert loaded_data == test_data
        assert loaded_data["persistent"] == "data"
        assert loaded_data["number"] == 42
        
        store2.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests
    test_memory_store_creation()
    print("✓ test_memory_store_creation passed")
    
    test_memory_store_save_and_load()
    print("✓ test_memory_store_save_and_load passed")
    
    test_memory_store_load_nonexistent_key()
    print("✓ test_memory_store_load_nonexistent_key passed")
    
    test_memory_store_append_to_list()
    print("✓ test_memory_store_append_to_list passed")
    
    test_memory_store_append_to_existing_list()
    print("✓ test_memory_store_append_to_existing_list passed")
    
    test_memory_store_search_by_tag()
    print("✓ test_memory_store_search_by_tag passed")
    
    test_memory_store_overwrite_existing_key()
    print("✓ test_memory_store_overwrite_existing_key passed")
    
    test_memory_store_persistence()
    print("✓ test_memory_store_persistence passed")
    
    print("\n✅ All tests passed!")

