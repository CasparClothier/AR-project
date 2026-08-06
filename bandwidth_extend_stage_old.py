"""AudioSR bandwidth-extension stage.

AudioSR is invoked as a subprocess (its CLI), not imported directly, since it
lives in a separate venv (audiosr-venv) with a dependency stack that conflicts
with the main project environment (numpy<=1.23.5, torch==2.4.0, etc. — see
project notes on environment setup).

AudioSR performs best on short (~5s) inputs and degrades on longer ones, so
this stage chunks the input, processes each chunk independently, and
recombines with a short linear crossfade to avoid audible clicks at chunk
boundaries.

The chunking/crossfade logic is pure and independently testable (see
verify_bandwidth_stage.py) without needing AudioSR itself installed or running
— important given how much environment work it took to get AudioSR running
at all.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Pure helpers — chunking and recombination, no subprocess involved.
# Testable in isolation with synthetic arrays.
# --------------------------------------------------------------------------- #
def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks.

    Chunks overlap by `overlap_seconds` so the crossfade in `crossfade_concat`
    has material to blend. The final chunk always reaches exactly the end of
    the audio, even if shorter than a full chunk.
    """
    chunk_len = int(chunk_seconds * sr) 
    overlap_len = int(overlap_seconds * sr)
    hop = chunk_len - overlap_len
    if hop <= 0:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    if len(audio) <= chunk_len: 
        return [audio]

    chunks = []
    idx = 0
    while idx < len(audio): # propogate until the last chunk reaches the end of the audio
        end = min(idx + chunk_len, len(audio))
        chunks.append(audio[idx:end])
        if end == len(audio):
            break
        idx += hop
    return chunks


def crossfade_concat(
    chunks: list[np.ndarray],
    overlap_samples: int,
) -> np.ndarray:
    """Recombine processed chunks with a linear crossfade over the overlap
    region, avoiding the click/discontinuity a hard concatenation would cause.

    `overlap_samples` is in the OUTPUT sample rate's units, since AudioSR
    changes the sample rate (e.g. 44100 -> 48000) — the overlap region is
    proportionally longer in the resampled output.
    """
    if len(chunks) == 1:
        return chunks[0]

    overlap_samples = max(0, min(overlap_samples, min(len(c) for c in chunks) // 2)) # if overlap is too long, reduce it to half the shortest chunk and ensure it's non-negative

    result = chunks[0].copy()
    for chunk in chunks[1:]:
        if overlap_samples == 0:
            result = np.concatenate([result, chunk])
            continue # if overlap is zero, just concatenate without crossfade

        fade_out = np.linspace(1.0, 0.0, overlap_samples) # linear fade-out multiplier for the tail of the previous chunk
        fade_in = np.linspace(0.0, 1.0, overlap_samples) # linear fade-in multiplier for the head of the current chunk

        tail = result[-overlap_samples:] * fade_out # apply fade-out to the tail of the previous chunk
        head = chunk[:overlap_samples] * fade_in # apply fade-in to the head of the current chunk
        blended = tail + head

        result = np.concatenate([result[:-overlap_samples], blended, chunk[overlap_samples:]])

    return result


# --------------------------------------------------------------------------- #
# The Stage
# --------------------------------------------------------------------------- #
class BandwidthExtendStage:
    """Restoration stage wrapping AudioSR via subprocess.

    AudioSR's own model runs in a separate venv; this class shells out to its
    CLI entry point rather than importing it directly.
    """

    def __init__(
        self,
        audiosr_python: str,
        model_name: str = "basic",
        ddim_steps: int = 25,
        seed: int = 42,
        chunk_seconds: float = 5.0,
        overlap_seconds: float = 0.25,
        output_sr: int = 48000,
    ) -> None:
        self.name = "bandwidth_extend"
        self.audiosr_python = audiosr_python
        self.model_name = model_name
        self.ddim_steps = ddim_steps
        self.seed = seed
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.output_sr = output_sr

    def _run_audiosr_on_chunk(self, chunk: np.ndarray, sr: int, work_dir: Path) -> np.ndarray:
        """Write one chunk to disk, run AudioSR on it, load and return the result."""
        chunk_in_path = work_dir / "chunk_in.wav"
        chunk_out_dir = work_dir / "chunk_out"
        chunk_out_dir.mkdir(exist_ok=True)

        sf.write(chunk_in_path, chunk, sr)

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        #result = subprocess.run(
        #    [
        #        self.audiosr_executable,
        #        "-i", str(chunk_in_path),
        #        "-s", str(chunk_out_dir),
        #        "--model_name", self.model_name,
        #        "--ddim_steps", str(self.ddim_steps),
        #        "--seed", str(self.seed),
        #    ],
        #    capture_output=True,
        #    text=True,
        #    timeout = 300
        #)

        result = subprocess.run(
            [
                #r"C:\dev\venvs\audiosr-venv\Scripts\python.exe",
                self.audiosr_python,
                "-m", "audiosr",
                "-i", str(chunk_in_path),
                "-s", str(chunk_out_dir),
                "--model_name", self.model_name,
                "--ddim_steps", str(self.ddim_steps),
                "--seed", str(self.seed),
            ],
        #capture_output=True,
        text=True,
        timeout=600,
        env=env,
        )


        if result.returncode != 0:
            raise RuntimeError(
                f"AudioSR subprocess failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )

        # Output lands under a timestamped subfolder with a fixed suffix;
        # search rather than construct the path, since we don't control
        # the timestamp component.
        matches = glob.glob(str(chunk_out_dir / "**" / "*_AudioSR_Processed_48K.wav"), recursive=True)
        if not matches:
            raise RuntimeError(
                f"AudioSR reported success but no output file found under {chunk_out_dir}.\n"
                f"stdout: {result.stdout[-1000:]}"
            )

        out_audio, out_sr = sf.read(matches[0])
        if out_sr != self.output_sr:
            raise RuntimeError(f"Expected AudioSR output at {self.output_sr} Hz, got {out_sr} Hz")
        return out_audio

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        chunks = chunk_audio(audio, sr, self.chunk_seconds, self.overlap_seconds)

        work_root = Path(tempfile.mkdtemp(prefix="audiosr_stage_"))
        try:
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_work_dir = work_root / f"chunk_{i:03d}"
                chunk_work_dir.mkdir(parents=True, exist_ok=True)
                processed = self._run_audiosr_on_chunk(chunk, sr, chunk_work_dir)
                processed_chunks.append(processed)

            # Overlap length in OUTPUT sample-rate units (AudioSR resamples,
            # e.g. 44100 -> 48000, so the overlap region is proportionally longer)
            overlap_samples_out = int(self.overlap_seconds * self.output_sr)
            result = crossfade_concat(processed_chunks, overlap_samples_out)

            return result.astype(np.float32), self.output_sr
        finally:
            shutil.rmtree(work_root, ignore_errors=True)


__all__ = ["BandwidthExtendStage", "chunk_audio", "crossfade_concat"]