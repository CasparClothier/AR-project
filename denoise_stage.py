"""Classical spectral-subtraction denoising.

Estimates a noise magnitude profile from the quietest frames of the signal
(same 5th-percentile-RMS approach used by analyze_audio's noise_floor_db),
then subtracts that profile from every frame's magnitude spectrum, leaving
phase untouched. Uses over-subtraction plus a spectral floor (the standard
Berouti formulation) to suppress the noise more aggressively than a naive
1:1 subtraction while avoiding "musical noise" — the classic spectral-
subtraction artefact where isolated bins survive subtraction and appear as
random tonal blips, worse than the noise it removes.

Chosen deliberately as a classical DSP method rather than a pretrained deep
model: no external dependency, no GPU, runs in the existing environment,
and gives a genuine classical-vs-deep contrast point alongside AudioSR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa


# --------------------------------------------------------------------------- #
# Pure functions — testable independent of any file I/O or Stage machinery.
# --------------------------------------------------------------------------- #
def estimate_noise_profile(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    quiet_percentile: float = 5.0,
) -> np.ndarray:
    """Estimate the noise magnitude spectrum from the quietest frames.

    Returns a 1D array of shape (n_fft//2 + 1,) — the average magnitude
    spectrum across whichever frames fall in the bottom `quiet_percentile`
    of per-frame RMS. Same underlying idea as analyze_audio's noise-floor
    measurement, but returning the full spectral shape rather than a single
    dB number.
    """
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)

    frame_rms = np.sqrt(np.mean(magnitude**2, axis=0))
    threshold = np.percentile(frame_rms, quiet_percentile)
    quiet_frames = magnitude[:, frame_rms <= threshold]

    if quiet_frames.shape[1] == 0:
        # Degenerate case (e.g. constant-level signal): fall back to the
        # single quietest frame rather than an empty average.
        quiet_frames = magnitude[:, [np.argmin(frame_rms)]]

    return quiet_frames.mean(axis=1)


def spectral_subtract(
    audio: np.ndarray,
    sr: int,
    noise_profile: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    over_subtraction: float = 2.0,
    spectral_floor: float = 0.02,
) -> np.ndarray:
    """Subtract `noise_profile` from every frame's magnitude spectrum.

    Standard Berouti over-subtraction: multiply the noise estimate by
    `over_subtraction` before subtracting (removes more noise than a naive
    1:1 subtraction would), and clamp each bin to at least `spectral_floor`
    times its original magnitude (prevents bins from being subtracted to
    exactly zero, which is what produces musical-noise artefacts — isolated
    surviving bins with no floor beneath them).

    Phase is taken unmodified from the input signal; only magnitude is
    altered. This is standard for spectral subtraction — phase errors are
    far less perceptually significant than magnitude errors at this scale.
    """
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    noise_profile = noise_profile.reshape(-1, 1)  # broadcast across time frames
    subtracted = magnitude - over_subtraction * noise_profile

    floor = spectral_floor * magnitude
    cleaned_magnitude = np.maximum(subtracted, floor)

    cleaned_S = cleaned_magnitude * np.exp(1j * phase)
    return librosa.istft(cleaned_S, hop_length=hop_length, length=len(audio))


def denoise(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    quiet_percentile: float = 5.0,
    over_subtraction: float = 1.0,
    spectral_floor: float = 0.15,
) -> np.ndarray:
    """Full pipeline: estimate noise profile from the signal itself, then subtract it."""
    profile = estimate_noise_profile(audio, sr, n_fft, hop_length, quiet_percentile)
    return spectral_subtract(
        audio, sr, profile, n_fft, hop_length, over_subtraction, spectral_floor
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# The Stage
# --------------------------------------------------------------------------- #
@dataclass
class DenoiseStage:
    """Restoration stage: classical spectral-subtraction denoising."""

    n_fft: int = 2048
    hop_length: int = 512
    quiet_percentile: float = 5.0
    over_subtraction: float = 1.0
    spectral_floor: float = 0.15

    name: str = "denoise"

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        cleaned = denoise(
            audio, sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            quiet_percentile=self.quiet_percentile,
            over_subtraction=self.over_subtraction,
            spectral_floor=self.spectral_floor,
        )
        return cleaned, sr  # spectral subtraction never changes sample rate


__all__ = [
    "estimate_noise_profile",
    "spectral_subtract",
    "denoise",
    "DenoiseStage",
]