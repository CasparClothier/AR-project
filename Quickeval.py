import librosa
from evaluate import evaluate_pair

original, sr1 = librosa.load(r"C:\Users\caspa\OneDrive\Desktop\Dsci\AR project\data\corpus_a\A-rag-time-episode_jukebox-129854_001_00-00-30.wav", sr=None)
restored, sr2 = librosa.load(r"C:\Users\caspa\OneDrive\Desktop\Dsci\AR project\test_output.wav", sr=None)

# Resample original to match restored's sample rate (48000) if they differ,
# since evaluate_pair truncates to matching length but doesn't resample
if sr1 != sr2:
    original = librosa.resample(original, orig_sr=sr1, target_sr=sr2)
    sr1 = sr2

result = evaluate_pair(original, restored, sr1, pair_id="rag_time_before_after")
print(f"SI-SDR: {result.si_sdr_db:.2f} dB  (NOT a quality score here — no clean reference exists)")
print(f"LSD:    {result.lsd_db:.2f} dB  (measures how much the spectrum changed)")