import os
import requests
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

def download_with_wget(url: str, dest_dir: Path) -> bool:
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {dest_dir} using wget...")
        subprocess.run(["wget", "-c", "-P", str(dest_dir), url], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Wget Error: {e}")
        return False
    except FileNotFoundError:
        print("Wget is not installed. Please install wget to download this dataset.")
        return False

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

def universal_downloader(
    source_type: str, source_id: str, dest_dir: Path, auth_token: str | None = None) -> bool:
    if source_type == "hf":
        return download_from_huggingface(source_id, dest_dir, auth_token)
    elif source_type in ("wget", "direct", "zenodo"):
        return download_with_wget(source_id, dest_dir)
    else:
        print(f"Unknown source type: {source_type}")
        return False