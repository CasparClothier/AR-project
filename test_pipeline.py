import numpy as np
from sklearn import pipeline
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

 
@pytest.fixture # fixes for the sample rate used in tests
def sample_rate():
    return 44100
 
 
@pytest.fixture
def test_audio(sample_rate):
    rng = np.random.default_rng(0) 
    t = np.linspace(0, 2.0, int(sample_rate * 2.0), endpoint=False) # create a 2-second test audio signal with two sine waves and some noise
    sig = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1760 * t) 
    sig += 0.01 * rng.standard_normal(len(t)) # add some noise
    return sig.astype(np.float32) # return the test audio signal as a float32 numpy array because soundfile expects float32 for writing audio files
 
 
@pytest.fixture
def input_file(tmp_path, test_audio, sample_rate):
    path = tmp_path / "input.wav" # create a temporary input file for testing
    sf.write(path, test_audio, sample_rate) # write the test audio signal to the temporary input file using soundfile
    return str(path)
 
 
@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "output.wav")
 

def test_sample_rate_change_propagates(monkeypatch, input_file, output_path):
    import pipeline2
    class UpsampleStage:
        name = "bandwidth_extend"
        def process(self, audio, sr):
            new_sr = 48000
            resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
            return resampled, new_sr 
    monkeypatch.setattr(pipeline2, "build_stages", lambda cfg=None: [UpsampleStage()]) # simulate a pipeline with a single stage that upsamples audio to 48kHz
    monkeypatch.setattr(
        pipeline2, "build_router",
        lambda cfg=None: type("R", (), {"decide": lambda self, stats, mel_db: {"bandwidth_extend": True}})(), # simulate a router that always chooses to run the upsample stage
    )
    cfg = PipelineConfig(target_sr=None)
    report = run_pipeline(input_file, output_path, cfg)
    assert report["final_sr"] == 48000
    _, out_sr = librosa.load(output_path, sr=None)
    assert out_sr == 48000 # ensure that the output file gets any changes to the sample rate that the pipeline makes
 

def test_identity(input_file, output_path): # Checks that the pipeline does not modify the audio when no processing is specified
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

def test_silent_input(tmp_path, sample_rate):
    silent = np.zeros(sample_rate, dtype=np.float32)
    in_path = str(tmp_path / "silent.wav")
    out_path = str(tmp_path / "silent_out.wav")
    sf.write(in_path, silent, sample_rate)
    cfg = PipelineConfig(target_sr=None)
    report = run_pipeline(in_path, out_path, cfg)
    assert report["final_sr"] == sample_rate