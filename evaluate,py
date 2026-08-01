"""Evaluation harness for Corpus B (clean, processed) pairs.

Two metrics, chosen to catch different failure modes (see project discussion):
  - SI-SDR : time-domain, catches gross energy errors (silence, hallucinated
             content, wrong overall scale). Scale-invariant variant of SDR,
             robust to a pipeline stage changing overall loudness.
  - LSD    : frequency-domain, catches spectral-shape errors (missing or
             wrong high-frequency content). Exactly the right lens for
             bandwidth-extension quality.

A model that's good at reconstruction scores well on both; a model that's
just leaving the input alone will score poorly on both when the input was
degraded, which is the sanity check to run before any real restoration stage
exists — see `evaluate.py`'s __main__ block.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import librosa


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Higher is better.

    Projects `estimate` onto `reference` to find the best-fit scale factor
    first, so a pipeline stage that changes overall loudness isn't penalised
    for something that isn't actually a reconstruction error.
    """
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)

    if len(estimate) != len(reference):
        n = min(len(estimate), len(reference))
        reference, estimate = reference[:n], estimate[:n]

    # Remove DC offset — SI-SDR is defined relative to zero-mean signals
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    ref_energy = np.sum(reference**2)
    if ref_energy < 1e-10:
        return float("nan")  # reference is silence; SI-SDR undefined

    # Best-fit scalar projection of estimate onto reference
    optimal_scale = np.sum(estimate * reference) / ref_energy
    projection = optimal_scale * reference
    projection_energy = np.sum(projection**2)

    noise = estimate - projection
    noise_energy = np.sum(noise**2)

    if projection_energy < 1e-10:
        # Estimate captured essentially none of the reference signal
        # (e.g. estimate is silence, or uncorrelated with reference).
        # This is a TOTAL FAILURE, not a perfect match — distinct from the
        # noise_energy check below.
        return float("-inf")
    if noise_energy < 1e-10:
        return float("inf")  # real signal captured AND negligible residual noise

    return 10 * np.log10(projection_energy / noise_energy)


def log_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """Mean log-spectral distance in dB. Lower is better; 0 is a perfect match.

    Computed on magnitude spectrograms only (phase is discarded, matching
    the standard LSD definition) — appropriate here since we care whether
    spectral *content* was reconstructed, not sample-exact phase alignment.
    """
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)

    if len(estimate) != len(reference):
        n = min(len(estimate), len(reference))
        reference, estimate = reference[:n], estimate[:n]

    ref_spec = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length))
    est_spec = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop_length))

    eps = 1e-10  # floor to avoid log(0) on silent bins
    log_ref = 20 * np.log10(ref_spec + eps)
    log_est = 20 * np.log10(est_spec + eps)

    # Per-frame RMS of the log-magnitude difference, then averaged over time
    frame_distance = np.sqrt(np.mean((log_ref - log_est) ** 2, axis=0))
    return float(np.mean(frame_distance))


# --------------------------------------------------------------------------- #
# Per-pair evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult: # data class to hold the evaluation results for a single pair of audio files (clean and processed). It contains the following fields:
    pair_id: str
    si_sdr_db: float
    lsd_db: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_pair( 
    clean: np.ndarray,
    processed: np.ndarray,
    sr: int,
    pair_id: str = "",
) -> EvaluationResult:
    """Compute both metrics for one (clean, processed) pair."""
    return EvaluationResult(
        pair_id=pair_id,
        si_sdr_db=si_sdr(clean, processed),
        lsd_db=log_spectral_distance(clean, processed, sr),
    )


# --------------------------------------------------------------------------- #
# Batch harness
# --------------------------------------------------------------------------- #
def evaluate_directory(
    clean_dir: str,
    processed_dir: str,
    sr: int | None = None,
) -> list[EvaluationResult]:
    """Evaluate every matching filename in clean_dir / processed_dir.

    Files are matched by name, so `corpus_b/clean/track01.wav` pairs with
    `corpus_b/processed/track01.wav`. Files present in one directory but not
    the other are skipped with a note printed to stdout, not a crash —
    a missing pair shouldn't abort an entire batch run.
    """
    clean_dir = Path(clean_dir)
    processed_dir = Path(processed_dir)

    clean_files = {f.name: f for f in clean_dir.glob("*.wav")}
    processed_files = {f.name: f for f in processed_dir.glob("*.wav")}

    common_names = sorted(set(clean_files) & set(processed_files))
    missing = (set(clean_files) | set(processed_files)) - set(common_names)
    if missing:
        print(f"Skipping {len(missing)} unmatched file(s): {sorted(missing)}")

    results = []
    for name in common_names:
        clean, clean_sr = librosa.load(clean_files[name], sr=sr)
        processed, proc_sr = librosa.load(processed_files[name], sr=sr)
        if clean_sr != proc_sr:
            # Resample processed to match clean's rate before comparing —
            # a stage may legitimately change sr (e.g. bandwidth extension)
            processed = librosa.resample(processed, orig_sr=proc_sr, target_sr=clean_sr)
        results.append(evaluate_pair(clean, processed, clean_sr, pair_id=name))

    return results


def summarise(results: list[EvaluationResult]) -> dict[str, float]:
    """Aggregate stats across a batch. Use median, not mean — historical-
    recording metrics can have outliers (a totally silent or corrupt file)
    that skew a mean badly."""
    sdrs = [r.si_sdr_db for r in results if np.isfinite(r.si_sdr_db)]
    lsds = [r.lsd_db for r in results if np.isfinite(r.lsd_db)]
    return {
        "n_pairs": len(results),
        "si_sdr_median": float(np.median(sdrs)) if sdrs else float("nan"),
        "si_sdr_mean": float(np.mean(sdrs)) if sdrs else float("nan"),
        "lsd_median": float(np.median(lsds)) if lsds else float("nan"),
        "lsd_mean": float(np.mean(lsds)) if lsds else float("nan"),
    }


def write_csv(results: list[EvaluationResult], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "si_sdr_db", "lsd_db"])
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


__all__ = [
    "si_sdr",
    "log_spectral_distance",
    "EvaluationResult",
    "evaluate_pair",
    "evaluate_directory",
    "summarise",
    "write_csv",
]