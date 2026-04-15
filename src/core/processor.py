import librosa
import soundfile as sf
from pathlib import Path
import concurrent.futures


def process_single_file(
    audio_path: Path, output_path: Path, target_sr: int | None = None, force_mono: bool = False
) -> bool:
    try:
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=force_mono)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, target_sr)

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def batch_process_audio(
    file_list: list[Path], output_dir: Path, target_sr: int = 16000, force_mono: bool = True, delete_originals: bool =False
):
    max_cpus_to_use = 4  # TODO: make this value dynamic
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus_to_use) as executor:
        futures = {
            executor.submit(
                process_single_file, audio_path, output_dir / audio_path.name, target_sr, force_mono
            ): audio_path
            for audio_path in file_list
        }

        for future in concurrent.futures.as_completed(futures):
            future.result()  # TODO: TUI progress bar

    if delete_originals:
        input_dir = file_list[0].parent 
        print(f"Cleanup: Deleting original folder {input_dir}")
        shutil.rmtree(input_dir)
