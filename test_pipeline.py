import numpy as np
import soundfile as sf
import librosa
import pytest

from pipeline2 import (
    PipelineConfig,
    NoOpStage,
    NoOpRouter,
    build_router,
    build_stages,
    run_pipeline,
    analyze_audio,

)

def test_sample_rate_change_propagates(monkeypatch, input_file, output_path):
    import pipeline
    class UpsampleStage:
        name = "bandwidth_extend"
        def process(self, audio, sr):
            new_sr = 48000
            resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
            return resampled, new_sr 
    monkeypatch.setattr(pipeline, "build_stages", lambda cfg=None: [UpsampleStage()]) # simulate a pipeline with a single stage that upsamples audio to 48kHz
    monkeypatch.setattr(
        pipeline, "build_router",
        lambda cfg=None: type("R", (), {"decide": lambda self, stats, mel_db: {"bandwidth_extend": True}})(), # simulate a router that always chooses to run the upsample stage
    )
    cfg = PipelineConfig(target_sr=None)
    report = run_pipeline(input_file, output_path, cfg)
    assert report["final_sr"] == 48000
    _, out_sr = librosa.load(output_path, sr=None)
    assert out_sr == 48000 # ensure that the output file gets any changes to the sample rate that the pipeline makes
 

def identity_test(input_file, output_path): # Checks that the pipeline does not modify the audio when no processing is specified
    cfg = PipelineConfig(target_sr=None)
    run_pipeline(input_file, output_path, cfg)
    in_audio, in_sr = librosa.load(input_file, sr=None)
    out_audio, out_sr = librosa.load(output_path, sr=None) # Will also ensure output file is created
    assert in_sr == out_sr
    assert in_audio.shape == out_audio.shape
    assert np.allclose(in_audio, out_audio, atol=1e-4)

def test_disabled_stage():
    calls = []
    class SpyStage:
        name = "declip"
        def process(self, audio, sr):
            calls.append(self.name)
            return audio, sr
    decisions = {"declip": False}
    stage = SpyStage()
    audio, sr = np.zeros(10), 44100
    if decisions.get(stage.name, False):
        audio, sr = stage.process(audio, sr)
    assert calls == []
 
 
def test_enabled_stage():
    calls = []
    class SpyStage:
        name = "declip"
        def process(self, audio, sr):
            calls.append(self.name)
            return audio, sr
    decisions = {"declip": True}
    stage = SpyStage()
    audio, sr = np.zeros(10), 44100
    if decisions.get(stage.name, False):
        audio, sr = stage.process(audio, sr)
    assert calls == ["declip"]

def silent_input(tmp_path, sample_rate):
    silent = np.zeros(sample_rate, dtype=np.float32)
    in_path = str(tmp_path / "silent.wav")
    out_path = str(tmp_path / "silent_out.wav")
    sf.write(in_path, silent, sample_rate)
    cfg = PipelineConfig(target_sr=None)
    report = run_pipeline(in_path, out_path, cfg)
    assert report["final_sr"] == sample_rate