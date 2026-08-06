import time
import librosa
import soundfile as sf
from bandwidth_extend_stage import BandwidthExtendStage

# Pick a real degraded acoustic-profile clip from your Corpus B
input_path = r"data\corpus_b\degraded\A_Classic_Education_-_NightOwl__acoustic.wav"

audio, sr = librosa.load(input_path, sr=None)
print(f"Input: {len(audio)/sr:.2f}s at {sr} Hz")

stage = BandwidthExtendStage(
    audiosr_python=r"C:\dev\venvs\audiosr-venv\Scripts\python.exe",
)

start = time.time()
result, out_sr = stage.process(audio, sr)
elapsed = time.time() - start

print(f"Output: {len(result)/out_sr:.2f}s at {out_sr} Hz")
print(f"Total time: {elapsed:.1f}s")

sf.write("stage_test_output.wav", result, out_sr)
print("Saved to stage_test_output.wav")