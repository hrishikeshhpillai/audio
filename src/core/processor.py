import torch
import torchaudio
from pathlib import Path
import concurrent.futures

def process_single_file(audio_path: Path, output_path: Path, target_sr: int = 16000, force_mono: bool = True) -> bool:
    try:
        waveform, sr = torchaudio.load(audio_path)

        if sr != target_sr:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resample(waveform)
        if force_mono:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, waveform, target_sr)

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def batch_process_audio(file_list: list[Path], output_dir: Path, target_sr: int = 16000):
    max_cpus_to_use = 4 #TODO: make this value dynamic
    with concurrent.futures.ProcessPoolExcecutor(max_workers=max_cpus_to_use) as executor:
        futures = {
            executor.submit(
                process_single_file,
                audio_path,
                output_dir / file_path.name
                target_sr
            ): audio_path for audio_path in file_list
        }

        for futures in concurrent.futures.as_completed(futures):
            future.result() #TODO: TUI progress bar
            