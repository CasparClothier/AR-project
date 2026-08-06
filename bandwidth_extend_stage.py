"""AudioSR bandwidth-extension stage — batched version.

AudioSR is invoked via its `-il` (input file list) batch mode, which loads
the model ONCE and processes every listed file within that single process.
This matters enormously on CPU: model loading takes ~5-6 minutes, dwarfing
the ~60s/chunk sampling cost. An earlier per-chunk-subprocess design reloaded
the model for every chunk (~6-7 min/chunk), making a 60-file corpus batch
run take 36+ hours. Batching brings a 30s file (6 chunks) down to roughly
one model load + 6x sampling time (~11-12 min), and a 60-file corpus to
roughly 11-12 hours — workable as an overnight run.

AudioSR performs best on short (~5s) inputs and degrades on longer ones, so
this stage chunks the input, processes all chunks for one file in a single
batched subprocess call, and recombines with a short linear crossfade to
avoid audible clicks at chunk boundaries.
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
# Testable in isolation with synthetic arrays (see verify_bandwidth_stage.py).
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
    while idx < len(audio):
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

    overlap_samples = max(0, min(overlap_samples, min(len(c) for c in chunks) // 2))

    result = chunks[0].copy()
    for chunk in chunks[1:]:
        if overlap_samples == 0:
            result = np.concatenate([result, chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, overlap_samples)
        fade_in = np.linspace(0.0, 1.0, overlap_samples)

        tail = result[-overlap_samples:] * fade_out
        head = chunk[:overlap_samples] * fade_in
        blended = tail + head

        result = np.concatenate([result[:-overlap_samples], blended, chunk[overlap_samples:]])

    return result


def find_processed_output(out_dir: Path, chunk_stem: str) -> Path:
    """Locate AudioSR's output file for a given input chunk stem.

    AudioSR writes to a TIMESTAMPED SUBFOLDER under out_dir with a fixed
    suffix pattern: '<stem>_AudioSR_Processed_48K.wav'. We search rather
    than construct the path since we don't control the timestamp component.
    """
    matches = glob.glob(str(out_dir / "**" / f"{chunk_stem}_AudioSR_Processed_48K.wav"), recursive=True)
    if not matches:
        raise RuntimeError(f"No AudioSR output found for chunk '{chunk_stem}' under {out_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous output for chunk '{chunk_stem}': found {len(matches)} matches under {out_dir}. "
            f"Use a fresh out_dir per file to avoid this."
        )
    return Path(matches[0])


# --------------------------------------------------------------------------- #
# The Stage
# --------------------------------------------------------------------------- #
class BandwidthExtendStage:
    """Restoration stage wrapping AudioSR via its batch-mode CLI.

    AudioSR's model runs in a separate venv; this class shells out to its
    Python interpreter directly (bypassing the .cmd/.exe entry-point wrapper,
    which resolves 'python' via the CALLING process's PATH rather than the
    target venv's own interpreter).
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
        timeout_seconds: int = 1800,
    ) -> None:
        self.name = "bandwidth_extend"
        self.audiosr_python = audiosr_python
        self.model_name = model_name
        self.ddim_steps = ddim_steps
        self.seed = seed
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.output_sr = output_sr
        self.timeout_seconds = timeout_seconds

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        chunks = chunk_audio(audio, sr, self.chunk_seconds, self.overlap_seconds)

        work_root = Path(tempfile.mkdtemp(prefix="audiosr_stage_"))
        try:
            # Write every chunk to disk, in order, with predictable names
            chunk_paths = []
            for i, chunk in enumerate(chunks):
                chunk_path = work_root / f"chunk_{i:03d}.wav"
                sf.write(chunk_path, chunk, sr)
                chunk_paths.append(chunk_path)

            # One batch list -> one subprocess call -> ONE model load
            batch_list_path = work_root / "batch.lst"
            batch_list_path.write_text("\n".join(str(p) for p in chunk_paths))

            out_dir = work_root / "batch_out"
            out_dir.mkdir()

            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"

            result = subprocess.run(
                [
                    self.audiosr_python, "-m", "audiosr",
                    "-il", str(batch_list_path),
                    "-s", str(out_dir),
                    "--model_name", self.model_name,
                    "--ddim_steps", str(self.ddim_steps),
                    "--seed", str(self.seed),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout_seconds,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"AudioSR batch subprocess failed (exit {result.returncode}):\n"
                    f"stdout: {result.stdout[-2000:]}\n"
                    f"stderr: {result.stderr[-2000:]}"
                )

            # Collect outputs IN THE SAME ORDER as chunk_paths — order matters,
            # since crossfade_concat blends adjacent chunks sequentially.
            processed_chunks = []
            for chunk_path in chunk_paths:
                out_path = find_processed_output(out_dir, chunk_path.stem)
                out_audio, out_sr = sf.read(out_path)
                if out_sr != self.output_sr:
                    raise RuntimeError(f"Expected {self.output_sr} Hz, got {out_sr} Hz for {chunk_path.name}")
                processed_chunks.append(out_audio)

            overlap_samples_out = int(self.overlap_seconds * self.output_sr)
            result_audio = crossfade_concat(processed_chunks, overlap_samples_out)

            return result_audio.astype(np.float32), self.output_sr
        finally:
            shutil.rmtree(work_root, ignore_errors=True)


__all__ = ["BandwidthExtendStage", "chunk_audio", "crossfade_concat", "find_processed_output"]