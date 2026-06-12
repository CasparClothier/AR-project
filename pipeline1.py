from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol
import librosa
import soundfile as sf


class Restoration(Protocol):
    name: str
    """Placeholder protocol for restoration stages."""
    def process(self, payload: Any, context: Optional[dict[str, Any]] = None) -> Any:
        ...


# class NoOpStage:
#    """A stage that passes data through unchanged."""

#    def __init__(self, name: str) -> None:
#        self.name = name

 #   def process(self, payload: Any, context: Optional[dict[str, Any]] = None) -> Any:
  #      return payload 

   # def __repr__(self) -> str:
    #    return f"<NoOpStage name={self.name!r}>"


class Pipeline:
    """A simple processing pipeline composed of stages."""

    def __init__(self, stages: Optional[Iterable[Stage]] = None) -> None: 
        self.stages: List[Stage] = list(stages) if stages is not None else []

    def add_stage(self, stage: Stage) -> None: 
        self.stages.append(stage)

    def validate(self) -> None: 
        if not self.stages:
            raise ValueError("Pipeline must contain at least one stage.")

        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline contains duplicate stage names.")

    def process(self, payload: Any, context: Optional[dict[str, Any]] = None) -> Any:
        self.validate()
        for stage in self.stages:
            payload = stage.process(payload, context=context)
        return payload


def build_noop_audio_pipeline() -> Pipeline:
    """Build a placeholder audio pipeline using noop stages."""
    return Pipeline(
        stages=[
            NoOpStage("capture"),
            NoOpStage("preprocessing"),
            NoOpStage("routing"),
            NoOpStage("restoration"),
            NoOpStage("postprocessing"),
            NoOpStage("delivery"),
        ]
    )


class NoOpRouter:
    """Router that always disables all processing paths."""

    def decide(self, stats: dict[str, Any], mel_db: Any) -> dict[str, bool]:
        return {"capture": False, "preprocessing": False, "routing": False, "restoration": False, "postprocessing": False, "delivery": False}


def build_router(config: Optional[dict[str, Any]] = None) -> NoOpRouter:
    return NoOpRouter()


def build_stages(config: Optional[Iterable[str]] = None) -> dict[str, NoOpStage]:
    # return mapping of stage name -> stage instance
    pipeline = build_noop_audio_pipeline()
    return {stage.name: stage for stage in pipeline.stages}


def analyze_audio(path: str, sr: int) -> dict[str, Any]:
    # lightweight placeholder analyzer
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)
    return {"duration": float(len(y) / sr), "rms": float((y**2).mean()), "mel_spectrogram_db": mel_db}


def loudness_normalise(audio: Any, sr: int, target_lufs: float) -> Any:
    # no-op placeholder
    return audio


def run_pipeline(input_path: str, output_path: str, config: Any) -> dict[str, Any]:
    audio, used_sr = librosa.load(input_path, sr=getattr(config, "target_sr", None))
    results = analyze_audio(input_path, sr=getattr(config, "target_sr", None))

    router = build_router(getattr(config, "routing", None))
    decisions = router.decide(stats=results, mel_db=results["mel_spectrogram_db"])

    stages = build_stages(getattr(config, "stages", None))
    for name, stage in stages.items():
        if decisions.get(name):
            # adapt to our NoOpStage.process signature
            audio = stage.process(audio)

    audio = loudness_normalise(audio, used_sr, getattr(config, "target_lufs", None))
    sf.write(output_path, audio, used_sr)

    return {
        "input": input_path,
        "output": output_path,
        "stats": {k: results[k] for k in results if k != "mel_spectrogram_db"},
        "decisions": decisions,
        "final_sr": used_sr,
    }

