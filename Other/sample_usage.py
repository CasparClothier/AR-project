from audio_quality import analyze_audio
import matplotlib.pyplot as plt


def demo(path: str):
    results = analyze_audio(path, sr=None)
    print("Clipping ratio:", results["clipping_ratio"])
    print("Noise floor (dB):", results["noise_floor_db"])
    print("Spectral cutoff (Hz):", results["spectral_cutoff_hz"])
    print("Duration (s):", results["duration_seconds"])

    S_db = results["mel_spectrogram_db"]
    try:
        import librosa.display

        librosa.display.specshow(S_db, sr=results["sr"], x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram (dB)")
        plt.tight_layout()
        plt.show()
    except Exception:
        plt.imshow(S_db, aspect="auto", origin="lower")
        plt.title("Mel spectrogram (dB)")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sample_usage.py /path/to/audio.wav")
    else:
        demo(sys.argv[1])
