"""Synthetic degradation for Corpus B (clean reference -> degraded input).

Applies three degradations, in the same order real damage tends to compound:
  1. Lowpass filter   (simulates bandwidth-limited transfer / narrow mic response)
  2. Clipping         (simulates overload during recording/transfer)
  3. Coloured noise   (simulates surface hiss / tape noise / broadcast noise)

Each degradation is parameterised by physically meaningful values (a cutoff
frequency, a clipping ratio, a noise floor in dB) rather than arbitrary knobs,
so a `DegradationProfile` can be built directly from `analyze_audio()`
measurements taken on real Corpus A recordings. That link is what makes the
synthetic degradation defensible as a proxy for real historical damage.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal


# --------------------------------------------------------------------------- #
# Degradation profile — the parameters, kept separate from the logic that
# applies them so a profile can be built by hand OR derived from measurements.
# --------------------------------------------------------------------------- #
@dataclass
class DegradationProfile:
    """Collect degradation parameters from analyze_audio() output into a single object,
      so they can be passed around"""
    spectral_cutoff_hz: float   # lowpass corner frequency
    clipping_ratio: float       # target fraction of samples clipped, 0.0-1.0
    noise_floor_db: float       # target noise floor, dB relative to full scale
    filter_order: int = 8       # steepness of the lowpass rolloff

    @classmethod
    def from_measurements(cls, stats: dict[str, float]) -> "DegradationProfile":
        """Build a profile directly from analyze_audio() output on a real
        Corpus A file, so the synthetic degradation matches measured reality."""
        return cls(
            spectral_cutoff_hz=stats["spectral_cutoff_hz"],
            clipping_ratio=stats["clipping_ratio"],
            noise_floor_db=stats["noise_floor_db"],
        )


# Named presets matching your two measured Corpus A files, so you can degrade
# Corpus B toward either profile without re-running analyze_audio every time.
PRESET_PARKER_BOOTLEG = DegradationProfile(
    spectral_cutoff_hz=7500.0,
    clipping_ratio=0.21,
    noise_floor_db=-40.0,
)

PRESET_ACOUSTIC_ERA = DegradationProfile(   # e.g. the Jukebox / LoC file
    spectral_cutoff_hz=3500.0,
    clipping_ratio=0.0,
    noise_floor_db=-30.0,
)


# --------------------------------------------------------------------------- #
# Individual degradation operations — each is independently testable
# --------------------------------------------------------------------------- #
def apply_lowpass(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    """Butterworth lowpass. Models bandwidth-limited historical transfers."""
    if cutoff_hz >= sr / 2:
        return audio.copy()  # nothing to do, cutoff is above Nyquist so no filtering occurs
    sos = signal.butter(order, cutoff_hz, btype="low", fs=sr, output="sos") # Design a digital Butterworth filter and return it in second-order sections (sos) format to minimise floating-point errors. The order of the filter determines the steepness of the rolloff, and the cutoff frequency is specified in Hz. The fs parameter specifies the sampling frequency of the audio signal.
    return signal.sosfiltfilt(sos, audio).astype(np.float32) # Apply the filter to the audio signal using zero-phase filtering (sosfiltfilt) to avoid phase distortion. The filtered audio is returned as a float32 array.


def apply_clipping(audio: np.ndarray, target_ratio: float) -> np.ndarray:
    """Hard-clip by finding the threshold that clips `target_ratio` of samples,
    then applying gain so the loudest samples cross it. Models overload."""
    if target_ratio <= 0.0: # if no clipping is desired, return the original audio
        return audio.copy()
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio.copy()  # silence, nothing to clip
    # Threshold amplitude below which (1 - target_ratio) of samples fall
    threshold = np.percentile(np.abs(audio), 100 * (1 - target_ratio))
    if threshold <= 0:
        return audio.copy()
    gain = peak / threshold  # push the target percentile up to clip at `peak`
    boosted = audio * gain
    return np.clip(boosted, -peak, peak).astype(np.float32)


def apply_noise(audio: np.ndarray, sr: int, target_noise_floor_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add broadband coloured noise (slight low-frequency emphasis, matching
    the measured surface-hiss profile) at a target RMS level."""
    noise = rng.standard_normal(len(audio)).astype(np.float32) # generate a standard normal distribution of random noise samples with the same length as the input audio signal. The noise is cast to float32 for consistency with the audio data type.
    # Mild low-frequency emphasis: a gentle 1-pole lowpass on the noise itself
    b, a = signal.butter(1, 4000, btype="low", fs=sr) # apply a 1st-order Butterworth lowpass filter with a cutoff frequency of 4000 Hz to the noise signal. This will emphasize lower frequencies in the noise, making it sound more like surface hiss or tape noise.
    noise = signal.lfilter(b, a, noise).astype(np.float32) # linear filter as opposed to zero-phase filtering, since phase distortion is not a concern for noise.
    target_rms = 10 ** (target_noise_floor_db / 20) # convert the target noise floor from dB to linear RMS amplitude. 
    noise_rms = np.sqrt(np.mean(noise**2)) 
    if noise_rms > 0: 
        noise *= target_rms / noise_rms # scale the noise signal to achieve the desired RMS level. This ensures that the added noise has the correct amplitude relative to the audio signal.
    return (audio + noise).astype(np.float32)


# --------------------------------------------------------------------------- #
# Orchestration — order matches how real damage compounds: bandwidth loss
# happens at the transfer stage, clipping happens on top of that, noise is
# the ever-present floor underneath everything.
# --------------------------------------------------------------------------- #
def degrade(
    audio: np.ndarray,
    sr: int,
    profile: DegradationProfile,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a full degradation profile to clean audio. Deterministic if `seed` given."""
    rng = np.random.default_rng(seed)

    out = apply_lowpass(audio, sr, profile.spectral_cutoff_hz, profile.filter_order) 
    out = apply_clipping(out, profile.clipping_ratio)
    out = apply_noise(out, sr, profile.noise_floor_db, rng)

    # Guard against inter-stage overflow before writing
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak

    return out.astype(np.float32)


__all__ = [
    "DegradationProfile",
    "PRESET_PARKER_BOOTLEG",
    "PRESET_ACOUSTIC_ERA",
    "apply_lowpass",
    "apply_clipping",
    "apply_noise",
    "degrade",
]