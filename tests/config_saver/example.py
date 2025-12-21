# MIT License
# Copyright (c) 2025 Anton Schreiner

import requests
from datetime import datetime
from typing import Any


class ConfigClient:
    """Client for interacting with the ML Config Tracker API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def save(self, name: str, config: dict[str, Any], overwrite: bool = False) -> dict:
        """
        Save a configuration.
        
        Args:
            name: Unique name for this config (e.g., "model-run-20-09-2022")
            config: Dictionary containing the configuration
            overwrite: If True, delete existing config with same name first
            
        Returns:
            Response from the server
        """
        if overwrite:
            # Try to delete existing config first (ignore if not found)
            try:
                self.delete(name)
            except requests.HTTPError:
                pass
        
        response = requests.post(
            f"{self.base_url}/api/configs",
            json={"name": name, "config": config}
        )
        response.raise_for_status()
        return response.json()
    
    def save_or_skip(self, name: str, config: dict[str, Any]) -> dict | None:
        """
        Save a configuration, but skip silently if it already exists.
        
        Returns:
            Response from the server, or None if config already exists
        """
        try:
            return self.save(name, config)
        except requests.HTTPError as e:
            if e.response.status_code == 409:
                return None
            raise
    
    def get(self, name: str) -> dict[str, Any]:
        """Get a configuration by name."""
        response = requests.get(f"{self.base_url}/api/configs/{name}")
        response.raise_for_status()
        return response.json()["config"]
    
    def list(self, search: str = "") -> list[dict]:
        """List all configurations, optionally filtered by search query."""
        response = requests.get(
            f"{self.base_url}/api/configs",
            params={"search": search} if search else {}
        )
        response.raise_for_status()
        return response.json()
    
    def delete(self, name: str) -> dict:
        """Delete a configuration by name."""
        response = requests.delete(f"{self.base_url}/api/configs/{name}")
        response.raise_for_status()
        return response.json()
    
    def diff(self, config1: str, config2: str) -> dict:
        """Compare two configurations."""
        response = requests.get(
            f"{self.base_url}/api/diff",
            params={"config1": config1, "config2": config2}
        )
        response.raise_for_status()
        return response.json()


# Convenience function for quick saves
def save_run_config(config: dict[str, Any], name: str | None = None, base_url: str = "http://localhost:8000") -> str:
    """
    Quick helper to save a training run configuration.
    
    Args:
        config: Your training configuration dict
        name: Optional name. If not provided, generates one with timestamp.
        base_url: URL of the config tracker server
        
    Returns:
        The name used to save the config
    """
    if name is None:
        name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    client = ConfigClient(base_url)
    client.save(name, config)
    return name


# Example usage
if __name__ == "__main__":
    # Example training config
    training_config = {
        "model": {
            "architecture": "transformer",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1,
        },
        "training": {
            "learning_rate": 3e-4,
            "batch_size": 32,
            "epochs": 100,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
        },
        "data": {
            "dataset": "imagenet",
            "augmentation": True,
            "preprocessing": "standard",
        },
        "optimizer": "adamw",
        "scheduler": "polynomial",
    }
    
    # Save with auto-generated name (always unique due to timestamp)
    run_name = save_run_config(training_config)
    print(f"Saved config as: {run_name}")
    
    # Or use the full client
    client = ConfigClient()
    
    # Save with custom name (skip if already exists)
    result = client.save_or_skip("experiment-baseline-v1", training_config)
    if result:
        print("Saved experiment-baseline-v1")
    else:
        print("experiment-baseline-v1 already exists, skipped")
    
    # Or overwrite existing config with same name
    # client.save("experiment-baseline-v1", training_config, overwrite=True)
    
    # List all configs
    configs = client.list()
    print(f"Found {len(configs)} configs")
    
    # Search for specific configs
    matching = client.list(search="baseline")
    print(f"Found {len(matching)} matching configs")