"""Modular audio-restoration pipeline (skeleton).

Flow:  load -> analyse -> route -> conditional restoration stages -> normalise -> write

The two extension points are:
   Router   - decides which restoration stages run (threshold-based or learned classifier)
   Stage    - performs one restoration step (declip, denoise, bandwidth-extend)

Both are Protocols, so any object with the right method shape satisfies them;
no explicit inheritance required. Concrete implementations (A-SPADE declipper,
AudioSR bandwidth extender, trained classifier router, ...) drop in later without
touching `run_pipeline`.
"""

from __future__ import annotations

from bandwidth_extend_stage import BandwidthExtendStage
from denoise_stage import DenoiseStage
from audio_quality import analyze_audio
from dataclasses import dataclass, field
from typing import Any, Protocol

import librosa
import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
# Define Protocols for stages and routers.
class Stage(Protocol):
    """A restoration step: transforms audio, may change the sample rate."""

    name: str

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]: 
        ...


class Router(Protocol):
    """Decides which restoration stages should run for a given recording."""

    def decide(self, stats: dict[str, Any], mel_db: np.ndarray) -> dict[str, bool]:
        ...


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass # Includes generated __init__, __repr__, and other methods; fields are defined as class attributes.
class PipelineConfig: # Holds all config for a pipeline run; passed to all components so they can adapt if needed.
    target_sr: int | None = None          # None preserves the file's native rate
    target_lufs: float = -23.0
    routing: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# No-op implementations (for testing / demo purposes, to be replaced by real logic later)
# --------------------------------------------------------------------------- #
class NoOpStage:
    """A restoration step: transforms audio, may change the sample rate."""
    

    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return audio, sr

    def __repr__(self) -> str:
        return f"<NoOpStage name={self.name!r}>"


class NoOpRouter:

    """Decides which restoration stages should run for a given recording."""

    STAGE_NAMES = ["declip", "denoise", "bandwidth_extend"]

    def decide(self, stats: dict[str, Any], mel_db: np.ndarray) -> dict[str, bool]:
        return {name: True for name in self.STAGE_NAMES}

    def __repr__(self) -> str:
        return "<NoOpRouter (enables none)>"



@dataclass
class ThresholdRouter:
    """Router that activates stages by comparing measured stats to fixed thresholds.

    Defaults set from the Corpus A catalogue (30 recordings, LoC National
    Jukebox collection): clipping never occurs on real data (median 0.0),
    spectral cutoff is uniformly severe (median ~2191 Hz), noise floor is
    the one axis with genuine spread (median ~-24.9 dB, range -33.6 to -21.5 dB).
    """

    clipping_ratio_threshold: float = 0.005
    spectral_cutoff_threshold_hz: float = 12000.0
    noise_floor_threshold_db: float = -28.0

    name = "threshold_router"

    def decide(self, stats: dict[str, Any], mel_db: np.ndarray) -> dict[str, bool]:
        return {
            "declip": stats["clipping_ratio"] > self.clipping_ratio_threshold,
            "denoise": stats["noise_floor_db"] > self.noise_floor_threshold_db,
            "bandwidth_extend": stats["spectral_cutoff_hz"] < self.spectral_cutoff_threshold_hz,
        }

    def __repr__(self) -> str:
        return (
            f"<ThresholdRouter clip>{self.clipping_ratio_threshold} "
            f"noise>{self.noise_floor_threshold_db}dB "
            f"cutoff<{self.spectral_cutoff_threshold_hz}Hz>"
        )



# --------------------------------------------------------------------------- #
# Factories — swap to introduce real routers / stages later
# --------------------------------------------------------------------------- #
#def build_router(config: dict[str, Any] | None = None) -> Router:
    # later: dispatch on config["type"] -> ThresholdRouter / ClassifierRouter
#    return NoOpRouter()

def build_router(config=None):
    config = config or {}
    if config.get("type") == "threshold":
        return ThresholdRouter(
            clipping_ratio_threshold=config.get("clipping_ratio_threshold", 0.005),
            noise_floor_threshold_db=config.get("noise_floor_threshold_db", -28.0),
            spectral_cutoff_threshold_hz=config.get("spectral_cutoff_threshold_hz", 12000.0),
        )
    return NoOpRouter()

#def build_stages(config: dict[str, Any] | None = None) -> list[Stage]:
    # later: dispatch on config -> DeclipStage / DenoiseStage / BandwidthExtendStage
    # Order matters: declip (fix waveform) -> denoise -> bandwidth_extend (fill spectrum)
#    return [
#        NoOpStage("declip"),
#        NoOpStage("denoise"),
#        NoOpStage("bandwidth_extend"),
#    ]

def build_stages(config=None):
    config = config or {}
    return [
        NoOpStage("declip"),
        #DenoiseStage(),
        NoOpStage("denoise"),
        BandwidthExtendStage(
            audiosr_python=config.get(
                "audiosr_python",
                r"C:\dev\venvs\audiosr-venv\Scripts\python.exe",
            ),
        ),
    ]

# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def loudness_normalise(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    # no-op placeholder; real impl uses pyloudnorm (EBU R128)
    return audio


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(input_path: str, output_path: str, config: PipelineConfig) -> dict[str, Any]:
    # 1. Load- check next step to ensure it is not being loaded twice.
    audio, sr = librosa.load(input_path, sr=config.target_sr) 

    # 2. Analyse
    results = analyze_audio(audio, sr)

    # 3. Route — decide which stages run
    router = build_router(config.routing)
    decisions = router.decide(stats=results, mel_db=results["mel_spectrogram_db"])

    # 4. Conditional restoration — only run activated stages
    stages = build_stages(config.stages)
    for stage in stages:
        if decisions.get(stage.name, False):
            audio, sr = stage.process(audio, sr)

    # 5. Always-on post-processing
    audio = loudness_normalise(audio, sr, config.target_lufs)

    # 6. Write
    sf.write(output_path, audio, sr)

    # 7. Report (exclude the large mel array from logged stats)
    return {
        "input": input_path,
        "output": output_path,
        "stats": {k: v for k, v in results.items() if k != "mel_spectrogram_db"},
        "decisions": decisions,
        "final_sr": sr,
    }


__all__ = [
    "Stage",
    "Router",
    "PipelineConfig",
    "NoOpStage",
    "NoOpRouter",
    "build_router",
    "build_stages",
    "analyze_audio",
    "loudness_normalise",
    "run_pipeline",
]