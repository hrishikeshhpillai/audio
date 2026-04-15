import pytest
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from src.core.processor import process_single_file, batch_process_audio


@pytest.fixture
def dummy_stereo_audio(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_stereo_48k_audio.wav"
    sample_rate = 48000
    array = np.random.randn(sample_rate, 2).astype(np.float32)
    sf.write(file_path, array, sample_rate)
    return file_path


@pytest.fixture
def dummy_mono_audio(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_mono_48k_audio.wav"
    sample_rate = 16000
    array = np.random.randn(sample_rate, 1).astype(np.float32)
    sf.write(file_path, array, sample_rate)
    return file_path


def test_process_single_file_downsample_and_forces_mono(dummy_stereo_audio: Path, tmp_path: Path):
    output_path = tmp_path / "processed_output.wav"
    target_sr = 16000
    success = process_single_file(dummy_stereo_audio, output_path, target_sr, True)

    assert success is True
    assert output_path.exists(), "Output file was not created"

    processed_file, processed_sr = sf.read(output_path)

    assert processed_sr == target_sr, f"Expected {target_sr}Hz, got {processed_sr}Hz"
    assert processed_file.ndim == 1, "Expected 1 Channel (Mono), but got Stereo"

def test_process_single_file_skips_processing_if_already_correct(
    dummy_mono_audio: Path, tmp_path: Path
):
    output_path = tmp_path / "processed_output.wav"
    target_sr = 16000

    success = process_single_file(dummy_mono_audio, output_path, target_sr, True)

    assert success is True

    processed_wave, processed_sr = sf.read(output_path)
    assert processed_sr == 16000
    assert processed_wave.ndim == 1


def test_process_single_file_missing_file_exception(tmp_path: Path):

    fake_input = tmp_path / "does_not_exist.wav"
    output_path = tmp_path / "output.wav"

    success = process_single_file(fake_input, output_path)

    assert success is False
    assert not output_path.exists()


def test_batch_process_audio_handles_multiple_files(tmp_path: Path):

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()

    file_paths = []
    for i in range(3):
        file_path = input_dir / f"test_audio_{i}.wav"
        array = np.random.randn(48000, 2).astype(np.float32)
        sf.write(file_path, array, 48000)
        file_paths.append(file_path)

    batch_process_audio(
        file_list=file_paths, output_dir=output_dir, target_sr=16000, force_mono=True, delete_originals=False
    )

    assert output_dir.exists(), "Output directory was not created"

    output_files = list(output_dir.glob("*.wav"))
    assert len(output_files) == 3, f"Expected 3 output files, found {len(output_files)}"

    for output_file in output_files:
        processed_file, processed_sr = sf.read(output_file)
        assert processed_sr == 16000, "File was not resampled correctly"
        assert processed_file.ndim == 1, "File was not forced to mono channel"
