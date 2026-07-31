"""Task 2: Build and analyse a catalogue of Corpus A degradation measurements.

Designed to be re-run cheaply as the corpus grows:
  - Skips files already in the catalogue (unless --force)
  - Appends new measurements rather than recomputing everything
  - Emits a distribution summary + clustering suggestion once you have enough files

Usage:
    python catalogue_corpus_a.py --input-dir data/corpus_a --out results/corpus_a_stats.csv
    python catalogue_corpus_a.py --input-dir data/corpus_a --out results/corpus_a_stats.csv --plot
    python catalogue_corpus_a.py --out results/corpus_a_stats.csv --analyse-only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import librosa

# Assumes audio_quality.analyze_audio takes (audio_array, sr) per the
# single-load refactor. Adjust the import if your module name differs.
from audio_quality import analyze_audio


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".m4a"}
CSV_FIELDS = [
    "filename",
    "duration_seconds",
    "sr",
    "clipping_ratio",
    "max_clipping_run_samples",
    "noise_floor_db",
    "spectral_cutoff_hz",
]


def find_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all audio files, sorted for deterministic ordering."""
    return sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    )


def load_existing_catalogue(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Read an existing catalogue so we can skip already-measured files."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def measure_file(path: Path) -> dict[str, Any] | None:
    """Run analyze_audio on one file. Returns None (with a note) on failure,
    so one corrupt file doesn't abort a whole batch run."""
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        stats = analyze_audio(audio, sr)
        return {
            "filename": path.name,
            "duration_seconds": round(stats["duration_seconds"], 3),
            "sr": stats["sr"],
            "clipping_ratio": round(stats["clipping_ratio"], 6),
            "max_clipping_run_samples": stats["max_clipping_run_samples"],
            "noise_floor_db": round(stats["noise_floor_db"], 2),
            "spectral_cutoff_hz": round(stats["spectral_cutoff_hz"], 1),
        }
    except Exception as exc:
        print(f"  ! FAILED {path.name}: {type(exc).__name__}: {exc}")
        return None


def build_catalogue(input_dir: Path, csv_path: Path, force: bool = False) -> list[dict[str, Any]]:
    existing = {} if force else load_existing_catalogue(csv_path)
    files = find_audio_files(input_dir)

    if not files:
        print(f"No audio files found in {input_dir}")
        return list(existing.values())

    print(f"Found {len(files)} audio file(s) in {input_dir}")
    if existing:
        print(f"  {len(existing)} already in catalogue, skipping (use --force to remeasure)")

    rows = list(existing.values())
    new_count = 0
    for path in files:
        if path.name in existing:
            continue
        print(f"  measuring {path.name} ...")
        row = measure_file(path)
        if row is not None:
            rows.append(row)
            new_count += 1

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCatalogue written to {csv_path} ({new_count} new, {len(rows)} total)")
    return rows


# --------------------------------------------------------------------------- #
# Distribution analysis — the point of the catalogue
# --------------------------------------------------------------------------- #
def describe_distribution(rows: list[dict[str, Any]]) -> None:
    """Print percentile summaries. Deliberately shows the SPREAD, not just
    means — the heterogeneity is the thing that motivates conditional routing."""
    if not rows:
        print("Catalogue is empty; nothing to describe.")
        return

    print(f"\n{'='*62}")
    print(f"DISTRIBUTION ACROSS {len(rows)} RECORDING(S)")
    print(f"{'='*62}")

    for field, label, fmt in [
        ("clipping_ratio", "Clipping ratio", "{:.4f}"),
        ("spectral_cutoff_hz", "Spectral cutoff (Hz)", "{:.0f}"),
        ("noise_floor_db", "Noise floor (dB)", "{:.1f}"),
    ]:
        values = np.array([float(r[field]) for r in rows])
        print(f"\n{label}:")
        print(f"  min    {fmt.format(np.min(values))}")
        print(f"  25th   {fmt.format(np.percentile(values, 25))}")
        print(f"  median {fmt.format(np.median(values))}")
        print(f"  75th   {fmt.format(np.percentile(values, 75))}")
        print(f"  max    {fmt.format(np.max(values))}")

        # Simple text histogram so you can see shape without opening a plot
        counts, edges = np.histogram(values, bins=min(8, max(3, len(values) // 2)))
        peak = counts.max() if counts.max() > 0 else 1
        print("  shape:")
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * int(20 * c / peak)
            print(f"    {fmt.format(lo):>10} - {fmt.format(hi):<10} |{bar} {c}")


def suggest_clusters(rows: list[dict[str, Any]], n_clusters: int = 3) -> None:
    """Cluster recordings by degradation profile to reveal natural groupings.

    Uses KMeans on standardised features. This is a SUGGESTION to look at,
    not an automatic decision — inspect the clusters and decide whether they
    correspond to real recording eras before building profiles from them.
    """
    if len(rows) < n_clusters * 2:
        print(f"\nNeed at least {n_clusters * 2} recordings to suggest {n_clusters} clusters "
              f"(have {len(rows)}). Collect more first.")
        return

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\nscikit-learn not installed; skipping clustering. "
              "Install with: pip install scikit-learn")
        return

    features = np.array([
        [float(r["clipping_ratio"]), float(r["spectral_cutoff_hz"]), float(r["noise_floor_db"])]
        for r in rows
    ])
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(scaled)

    print(f"\n{'='*62}")
    print(f"SUGGESTED DEGRADATION PROFILES ({n_clusters} clusters)")
    print(f"{'='*62}")
    print("Inspect these — do they map onto real recording eras?\n")

    for c in range(n_clusters):
        members = features[labels == c]
        names = [rows[i]["filename"] for i in range(len(rows)) if labels[i] == c]
        print(f"Cluster {c}  ({len(members)} recordings)")
        print(f"  median clipping ratio : {np.median(members[:, 0]):.4f}")
        print(f"  median spectral cutoff: {np.median(members[:, 1]):.0f} Hz")
        print(f"  median noise floor    : {np.median(members[:, 2]):.1f} dB")
        print(f"  -> DegradationProfile(spectral_cutoff_hz={np.median(members[:, 1]):.0f}, "
              f"clipping_ratio={np.median(members[:, 0]):.4f}, "
              f"noise_floor_db={np.median(members[:, 2]):.1f})")
        preview = ", ".join(names[:3]) + ("..." if len(names) > 3 else "")
        print(f"  members: {preview}\n")


def plot_distribution(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Save histograms + a scatter of the two most informative axes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    clipping = np.array([float(r["clipping_ratio"]) for r in rows])
    cutoff = np.array([float(r["spectral_cutoff_hz"]) for r in rows])
    noise = np.array([float(r["noise_floor_db"]) for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(clipping, bins=12, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Clipping ratio")
    axes[0, 1].hist(cutoff, bins=12, color="darkorange", edgecolor="black")
    axes[0, 1].set_title("Spectral cutoff (Hz)")
    axes[1, 0].hist(noise, bins=12, color="seagreen", edgecolor="black")
    axes[1, 0].set_title("Noise floor (dB)")
    axes[1, 1].scatter(cutoff, clipping, c=noise, cmap="viridis", s=60, edgecolor="black")
    axes[1, 1].set_xlabel("Spectral cutoff (Hz)")
    axes[1, 1].set_ylabel("Clipping ratio")
    axes[1, 1].set_title("Degradation space (colour = noise floor)")

    fig.suptitle(f"Corpus A degradation profile ({len(rows)} recordings)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nPlots saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Catalogue Corpus A degradation measurements.")
    ap.add_argument("--input-dir", type=Path, help="Directory of Corpus A recordings")
    ap.add_argument("--out", type=Path, default=Path("results/corpus_a_stats.csv"))
    ap.add_argument("--force", action="store_true", help="Remeasure all files")
    ap.add_argument("--analyse-only", action="store_true", help="Skip measuring; analyse existing CSV")
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--plot", action="store_true", help="Save distribution plots")
    args = ap.parse_args()

    if args.analyse_only:
        rows = list(load_existing_catalogue(args.out).values())
        if not rows:
            print(f"No catalogue found at {args.out}")
            return
    else:
        if not args.input_dir:
            ap.error("--input-dir is required unless using --analyse-only")
        rows = build_catalogue(args.input_dir, args.out, force=args.force)

    describe_distribution(rows)
    suggest_clusters(rows, n_clusters=args.clusters)

    if args.plot:
        plot_distribution(rows, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()