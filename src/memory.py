"""
Memory Store - Session and long-term memory management.

This module provides persistent storage for user progress, preferences,
and session data using TinyDB (preferred) or atomic JSON file writes (fallback).
"""

import json
import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path

# Try to import TinyDB, fallback to None if not available
try:
    from tinydb import TinyDB, Query
    TINYDB_AVAILABLE = True
except ImportError:
    TINYDB_AVAILABLE = False
    Query = None


# Schema version for migration support
CURRENT_SCHEMA_VERSION = "1.0"


class MemoryStore:
    """
    Memory store for persisting user data, progress, and session information.
    
    Supports both TinyDB (preferred) and atomic JSON file writes (fallback).
    """
    
    def __init__(self, storage_path: str = "data/memory_store.json", use_tinydb: Optional[bool] = None):
        """
        Initialize the memory store.
        
        Args:
            storage_path: Path to the storage file (JSON or TinyDB)
            use_tinydb: Force use of TinyDB (True) or JSON (False). 
                       If None, auto-detect based on availability.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine storage backend
        if use_tinydb is None:
            self.use_tinydb = TINYDB_AVAILABLE
        else:
            self.use_tinydb = use_tinydb and TINYDB_AVAILABLE
        
        if self.use_tinydb:
            self.db = TinyDB(str(self.storage_path))
            self._initialize_schema()
        else:
            self._initialize_json_store()
    
    def _initialize_schema(self) -> None:
        """Initialize database schema and migrate if needed."""
        if not self.use_tinydb:
            return
        
        # Check if schema version exists
        metadata = self.db.get(Query().key == "__metadata__")
        
        if metadata is None:
            # First time initialization
            self.db.insert({
                "key": "__metadata__",
                "value": {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "created_at": self._get_timestamp()
                }
            })
        else:
            # Check schema version and migrate if needed
            current_version = metadata.get("value", {}).get("schema_version", "0.0")
            if current_version != CURRENT_SCHEMA_VERSION:
                self._migrate_schema(current_version, CURRENT_SCHEMA_VERSION)
    
    def _initialize_json_store(self) -> None:
        """Initialize JSON storage file if it doesn't exist."""
        if not self.storage_path.exists():
            initial_data = {
                "__metadata__": {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "created_at": self._get_timestamp()
                }
            }
            self._atomic_json_write(initial_data)
    
    def _migrate_schema(self, from_version: str, to_version: str) -> None:
        """
        Migrate schema from one version to another.
        
        Args:
            from_version: Current schema version
            to_version: Target schema version
        """
        # Update metadata
        metadata = self.db.get(Query().key == "__metadata__")
        if metadata:
            metadata["value"]["schema_version"] = to_version
            metadata["value"]["migrated_at"] = self._get_timestamp()
            self.db.update(metadata, Query().key == "__metadata__")
        
        # Add version-specific migrations here as needed
        # Example: if from_version == "0.9" and to_version == "1.0":
        #     # Perform migration logic
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save(self, key: str, value: Any) -> None:
        """
        Save a value associated with a key.
        
        Args:
            key: Unique identifier for the data
            value: Data to store (must be JSON-serializable)
        """
        if self.use_tinydb:
            # Check if key exists
            existing = self.db.get(Query().key == key)
            if existing:
                self.db.update({"value": value}, Query().key == key)
            else:
                self.db.insert({"key": key, "value": value})
        else:
            data = self._load_json()
            data[key] = value
            self._atomic_json_write(data)
    
    def load(self, key: str, default: Any = None) -> Any:
        """
        Load a value by key.
        
        Args:
            key: Key to look up
            default: Default value if key doesn't exist
            
        Returns:
            Stored value or default if not found
        """
        if self.use_tinydb:
            result = self.db.get(Query().key == key)
            if result:
                return result.get("value", default)
            return default
        else:
            data = self._load_json()
            return data.get(key, default)
    
    def append_to_list(self, key: str, item: Any) -> None:
        """
        Append an item to a list stored at the given key.
        Creates the list if it doesn't exist.
        
        Args:
            key: Key of the list to append to
            item: Item to append
        """
        current_list = self.load(key, default=[])
        
        # Ensure it's a list
        if not isinstance(current_list, list):
            current_list = [current_list]
        
        current_list.append(item)
        self.save(key, current_list)
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Search for all entries that contain the given tag.
        
        Tags are expected to be in the format:
        {
            "tags": ["tag1", "tag2", ...],
            ...
        }
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching entries (as dictionaries with 'key' and 'value')
        """
        results = []
        
        if self.use_tinydb:
            # Search all entries
            all_entries = self.db.all()
            for entry in all_entries:
                key = entry.get("key", "")
                value = entry.get("value", {})
                
                # Skip metadata
                if key == "__metadata__":
                    continue
                
                # Check if value has tags
                if isinstance(value, dict) and "tags" in value:
                    if tag in value.get("tags", []):
                        results.append({"key": key, "value": value})
                # Also check if key contains the tag
                elif tag in key:
                    results.append({"key": key, "value": value})
        else:
            data = self._load_json()
            for key, value in data.items():
                if key == "__metadata__":
                    continue
                
                # Check if value has tags
                if isinstance(value, dict) and "tags" in value:
                    if tag in value.get("tags", []):
                        results.append({"key": key, "value": value})
                # Also check if key contains the tag
                elif tag in key:
                    results.append({"key": key, "value": value})
        
        return results
    
    def _load_json(self) -> Dict[str, Any]:
        """Load JSON data from file."""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted, return empty dict
            print(f"Warning: Could not load JSON from {self.storage_path}: {e}")
            return {}
    
    def _atomic_json_write(self, data: Dict[str, Any]) -> None:
        """
        Write JSON data atomically to prevent corruption.
        
        Writes to a temporary file first, then renames it to the target file.
        """
        # Create temporary file in the same directory
        temp_file = self.storage_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (works on Unix and Windows)
            if os.name == 'nt':  # Windows
                # On Windows, need to remove target first
                if self.storage_path.exists():
                    os.remove(self.storage_path)
                shutil.move(str(temp_file), str(self.storage_path))
            else:  # Unix/Linux
                shutil.move(str(temp_file), str(self.storage_path))
                
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to write to {self.storage_path}: {e}")
    
    def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Get all memory entries for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user's memory entries
        """
        user_key_prefix = f"user:{user_id}:"
        user_data = {}
        
        if self.use_tinydb:
            all_entries = self.db.all()
            for entry in all_entries:
                key = entry.get("key", "")
                if key.startswith(user_key_prefix):
                    # Remove prefix for cleaner keys
                    clean_key = key[len(user_key_prefix):]
                    user_data[clean_key] = entry.get("value")
        else:
            data = self._load_json()
            for key, value in data.items():
                if key.startswith(user_key_prefix):
                    clean_key = key[len(user_key_prefix):]
                    user_data[clean_key] = value
        
        return user_data
    
    def close(self) -> None:
        """Close the database connection (for TinyDB)."""
        if self.use_tinydb and hasattr(self, 'db'):
            self.db.close()


# Initialize sample memory for demo_user
def initialize_demo_user_memory(store: MemoryStore) -> None:
    """
    Initialize sample memory data for demo_user.
    
    Args:
        store: MemoryStore instance to populate
    """
    user_id = "demo_user"
    
    # User profile
    store.save(f"user:{user_id}:profile", {
        "user_id": user_id,
        "name": "Demo User",
        "level": "intermediate",
        "preferences": {
            "learning_style": "visual",
            "difficulty": "intermediate"
        },
        "tags": ["demo", "user", "profile"]
    })
    
    # Learning progress
    store.save(f"user:{user_id}:progress", {
        "topics_studied": ["linear_regression", "neural_networks"],
        "topics_mastered": ["linear_regression"],
        "current_topic": "neural_networks",
        "tags": ["demo", "progress", "learning"]
    })
    
    # Wrong answers log
    store.append_to_list(f"user:{user_id}:wrong_answers", {
        "topic": "linear_regression",
        "question": "What is the cost function for linear regression?",
        "student_answer": "Mean squared error",
        "correct_answer": "Mean squared error (MSE) or Mean absolute error (MAE)",
        "timestamp": store._get_timestamp(),
        "tags": ["wrong_answer", "linear_regression"]
    })
    
    # Session notes
    store.append_to_list(f"user:{user_id}:notes", {
        "topic": "neural_networks",
        "note": "Need to review backpropagation algorithm",
        "timestamp": store._get_timestamp(),
        "tags": ["note", "neural_networks", "review"]
    })


# Example usage
if __name__ == "__main__":
    # Create memory store
    store = MemoryStore(storage_path="data/memory_store.json")
    
    # Initialize demo user
    initialize_demo_user_memory(store)
    
    # Test save and load
    store.save("test_key", {"data": "test_value", "tags": ["test"]})
    print("Saved value:", store.load("test_key"))
    
    # Test append to list
    store.append_to_list("test_list", "item1")
    store.append_to_list("test_list", "item2")
    print("List after appends:", store.load("test_list"))
    
    # Test search by tag
    results = store.search_by_tag("demo")
    print(f"\nFound {len(results)} entries with tag 'demo':")
    for result in results:
        print(f"  - {result['key']}: {result['value']}")
    
    # Get user memory
    user_mem = store.get_user_memory("demo_user")
    print(f"\nDemo user memory keys: {list(user_mem.keys())}")
    
    # Close store
    store.close()
    print("\nMemory store operations completed successfully!")

