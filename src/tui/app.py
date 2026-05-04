import time
import yaml
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
CATALOG_PATH = Path("/catalog.yaml")

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
                    
                    if not repo_id or name or ds_type or source:
                        continue # Skip empty/invalid entries
                        
                    display_name = f"[{ds_type}] {name}"
                    
                    # Store as "source|repo_id"
                    datasets_map[display_name] = f"{source}|{repo_id}"
                    
        except Exception as e:
            console.print(f"[bold red]Error parsing catalog.yaml: {e}[/bold red]")

    datasets_map["Custom URL..."] = "custom"
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

    if DATASETS[dataset_choice] == "custom":
        try:
            source_val = inquirer.text(message="Enter the HuggingFace ID or URL:").execute()
            if not source_val: return
        except KeyboardInterrupt:
            return
            
        source_type = "hf" if "/" in source_val and "http" not in source_val else "direct"
        source_id = source_val
    else:
        source_type, source_id = DATASETS[dataset_choice].split("|")

    try:
        force_mono = inquirer.confirm(message="Force audio to Mono channel?", default=True).execute()
    except KeyboardInterrupt:
        return 
    
    console.print(f"\n[bold green]Starting pipeline for {source_id}...[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True, 
    ) as progress:
        
        task1 = progress.add_task("[cyan]Downloading dataset...", total=None)
        success = universal_downloader(source_type, source_id, RAW_DIR)
        
        if not success:
            progress.stop()
            console.print("[bold red] Download failed![/bold red] Check terminal for errors.")
            return
            
        progress.update(task1, completed=100, description="[green] Dataset downloaded![/green]")

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
            force_mono=force_mono
        )
        progress.update(task2, completed=100, description="[green]✔ Audio processed![/green]")

    console.print("\n[bold magenta]✨ Pipeline Complete! ✨[/bold magenta]")
    console.print(f"📁 Raw files:       [dim]{RAW_DIR.absolute()}[/dim]")
    console.print(f"📁 Processed files: [dim]{PROCESSED_DIR.absolute()}[/dim]\n")

if __name__ == "__main__":
    WORKSPACE_DIR.mkdir(exist_ok=True)
    run_wizard()