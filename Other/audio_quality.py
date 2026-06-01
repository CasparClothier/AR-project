import numpy as np
import librosa
from typing import Any, Dict


def _longest_true_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    idx = np.flatnonzero(mask)
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return max(len(s) for s in splits)


def analyze_audio(
    path: str,
    sr: int | None = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    energy_percentile: float = 0.95,
) -> Dict[str, Any]:
    """Load `path` and compute clipping, noise floor, spectral cutoff and a mel spectrogram.

    Returns a dict with keys:
      - `clipping_ratio` (float): fraction of samples clipped
      - `max_clipping_run_samples` (int): longest consecutive clipped run (samples)
      - `noise_floor_db` (float): estimated noise floor in dB (RMS percentile)
      - `spectral_cutoff_hz` (float): frequency below which `energy_percentile` of energy lies
      - `mel_spectrogram_db` (np.ndarray): mel spectrogram in dB (shape: n_mels x frames)
      - `sr` and `duration_seconds`

    Notes:
      - Uses `librosa` for reading and feature extraction.
      - `sr=None` preserves original sample rate.
    """
    y, used_sr = librosa.load(path, sr=sr)
    total_samples = y.shape[0]
    duration_seconds = float(total_samples) / used_sr

    # Clipping: count samples near +/-1.0
    clip_thresh = 0.999
    clipped_mask = np.abs(y) >= clip_thresh
    clipping_ratio = float(np.count_nonzero(clipped_mask)) / max(1, total_samples) # If no samples, avoid div by zero
    max_clipping_run = _longest_true_run(clipped_mask)

    # Noise floor: frame RMS, take low percentile (10th) as noise floor amplitude
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    noise_floor_amp = float(np.percentile(rms, 10))
    # Convert amplitude to dB (relative to 1.0)
    noise_floor_db = float(librosa.amplitude_to_db(np.array([noise_floor_amp]), ref=1.0)[0])

    # Spectral cutoff: compute mean power spectrum and find freq at percentile
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=used_sr, n_fft=n_fft)
    mean_spec = S.mean(axis=1)
    total_energy = float(mean_spec.sum()) if mean_spec.sum() > 0 else 1.0
    cum = np.cumsum(mean_spec) / total_energy
    cutoff_idx = int(np.searchsorted(cum, energy_percentile, side="left"))
    cutoff_idx = min(max(cutoff_idx, 0), len(freqs) - 1)
    spectral_cutoff_hz = float(freqs[cutoff_idx])

    # Mel spectrogram (power) -> dB
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=used_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    return {
        "clipping_ratio": clipping_ratio,
        "max_clipping_run_samples": int(max_clipping_run),
        "noise_floor_db": noise_floor_db,
        "spectral_cutoff_hz": spectral_cutoff_hz,
        "mel_spectrogram_db": S_mel_db,
        "sr": used_sr,
        "duration_seconds": duration_seconds,
    }


__all__ = ["analyze_audio"]
