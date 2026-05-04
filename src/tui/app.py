import time
import yaml
import shutil
from pathlib import Path
from InquirerPy import inquirer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.processor import batch_process_audio
from core.downloader import universal_downloader

console = Console()

WORKSPACE_DIR = Path("./wizard_workspace")
RAW_DIR = WORKSPACE_DIR / "raw"
PROCESSED_DIR = WORKSPACE_DIR / "processed"
CATALOG_PATH = Path("../../catalog.yaml")

def load_catalog(filepath: Path) -> dict:

    datasets_map = {}
    
    if not filepath.exists():
        console.print(f"[yellow]Warning: {filepath} not found. Creating a template.[/yellow]")
    else:
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
                
            if data and "datasets" in data:

                for ds in data["datasets"]:

                    name = ds.get("name", "")
                    ds_type = ds.get("type", "")
                    source = ds.get("source", "")
                    repo_id = ds.get("repo_id", "")
                    subsets = ds.get("subsets", [])
                    
                    if not repo_id or not name or not ds_type or not source:
                        continue # Skip empty/invalid entries
                        
                    display_name = f"[{ds_type}] {name}"
                    
                    # Store dataset info
                    datasets_map[display_name] = {"source": source, "repo_id": repo_id, "subsets": subsets}
                    
        except Exception as e:
            console.print(f"[bold red]Error parsing catalog.yaml: {e}[/bold red]")

    datasets_map["Custom URL..."] = {"source": "custom", "repo_id": "", "subsets": []}
    return datasets_map

def run_wizard():

    console.print("\n[bold cyan] Welcome to the Audio Wizard[/bold cyan]\n")

    # Load data dynamically from the new schema
    DATASETS = load_catalog(CATALOG_PATH)

    try:
        dataset_choice = inquirer.fuzzy(
            message="Search and select a dataset:",
            choices=list(DATASETS.keys()),
            instruction="(Start typing to filter, arrow keys to navigate)",
            max_height="50%"
        ).execute()
    except KeyboardInterrupt:
        console.print("[yellow]Wizard cancelled.[/yellow]")
        return

    if DATASETS[dataset_choice]["source"] == "custom":
        try:
            source_val = inquirer.text(message="Enter the HuggingFace ID or URL:").execute()
            if not source_val: return
        except KeyboardInterrupt:
            return
            
        source_type = "hf" if "/" in source_val and "http" not in source_val else "direct"
        source_id = source_val
    else:
        dataset_info = DATASETS[dataset_choice]
        source_type = dataset_info["source"]
        source_id = dataset_info["repo_id"]
        subsets = dataset_info.get("subsets", [])
        
        if subsets and len(subsets) > 0:
            try:
                subset_choice = inquirer.fuzzy(
                    message="Select a subset to download:",
                    choices=subsets,
                    instruction="(Start typing to filter, arrow keys to navigate)",
                    max_height="50%"
                ).execute()
                if source_type in ["wget", "direct", "zenodo"]:
                    if not source_id.endswith("/"):
                        source_id += "/"
                    source_id += subset_choice
            except KeyboardInterrupt:
                console.print("[yellow]Wizard cancelled.[/yellow]")
                return

    try:
        force_mono = inquirer.confirm(message="Force audio to Mono channel?", default=True).execute()
        
        fix_length = inquirer.confirm(message="Pad/Truncate audio to a fixed length?", default=False).execute()
        fixed_length_seconds = None
        if fix_length:
            length_input = inquirer.text(
                message="Enter fixed length in seconds (e.g. 5.0):",
                validate=lambda val: val.replace('.', '', 1).isdigit() and float(val) > 0,
                invalid_message="Please enter a valid positive number"
            ).execute()
            fixed_length_seconds = float(length_input)
            
    except KeyboardInterrupt:
        return 
    
    console.print(f"\n[bold green]Starting download for {source_id}...[/bold green]")
    try:
        success = universal_downloader(source_type, source_id, RAW_DIR)
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user. Cleaning up...[/yellow]")
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
        return
    
    if not success:
        console.print("[bold red] Download failed![/bold red] Check terminal for errors.")
        return
        
    console.print("[green] Dataset downloaded![/green]")

    console.print("\n[bold green]Starting audio processing...[/bold green]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True, 
    ) as progress:
        
        task2 = progress.add_task("[yellow]Processing audio...", total=None)
        
        files = list(RAW_DIR.rglob("*.wav"))
        if not files:
            progress.stop()
            console.print("[bold red] No .wav files found in download![/bold red]")
            return

        batch_process_audio(
            file_list=files, 
            output_dir=PROCESSED_DIR, 
            target_sr=16000, 
            force_mono=force_mono,
            fixed_length_seconds=fixed_length_seconds
        )
        progress.update(task2, completed=100, description="[green]✔ Audio processed![/green]")

    console.print("\n[bold magenta]✨ Pipeline Complete! ✨[/bold magenta]")
    console.print(f"📁 Raw files:       [dim]{RAW_DIR.absolute()}[/dim]")
    console.print(f"📁 Processed files: [dim]{PROCESSED_DIR.absolute()}[/dim]\n")

if __name__ == "__main__":
    WORKSPACE_DIR.mkdir(exist_ok=True)
    run_wizard()