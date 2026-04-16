import os
import requests
from pathlib import Path
from huggingface_hub import snapshot_download
import kaggle

def download_from_huggingface(repo_id: str, dest_dir: Path, auth_token: str | None = None) -> bool:
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest_dir,
            token=auth_token
        )
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        return False

def universal_download(
    source_type: str, source_id: str, dest_dir: Path, auth_token: str | None = None) -> bool:
    if source_type == "hf":
        return download_huggingface(source_id, dest_dir, auth_token)
    else:
        print(f"Unknown source type: {source_type}")
        return False