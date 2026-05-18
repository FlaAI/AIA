import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch

DEFAULT_WAV_DIR = ""
DEFAULT_NPZ_DIR = ""
DEFAULT_OUT_DIR = ""


def get_duration(wav_path: Path) -> Tuple[float, int]:
    info = torchaudio.info(str(wav_path))
    duration = info.num_frames / info.sample_rate if info.sample_rate > 0 else 0.0
    return duration, info.sample_rate


def summarize(durations: List[float]) -> dict:
    arr = np.array(durations)
    return {
        "count": int(arr.size),
        "min_sec": float(arr.min()) if arr.size else 0.0,
        "max_sec": float(arr.max()) if arr.size else 0.0,
        "mean_sec": float(arr.mean()) if arr.size else 0.0,
        "median_sec": float(np.median(arr)) if arr.size else 0.0,
        "p90_sec": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p95_sec": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99_sec": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }


def main(args):
    wav_dir = Path(args.wav_dir)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav_dir = out_dir / "wav"
    out_npz_dir = out_dir / "npz"
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    if args.max_files and args.max_files > 0:
        wav_files = wav_files[: args.max_files]

    durations = []
    sample_rates = []
    for wp in wav_files:
        dur, sr = get_duration(wp)
        durations.append(dur)
        sample_rates.append(sr)

    stats = summarize(durations)
    sr_set = set(sample_rates)
    over_5s = sum(d > 5.0 for d in durations)

    print(f"[*] Total wav files analyzed: {len(wav_files)}")
    print(f"[*] Sample rates observed: {sorted(sr_set)}")
    print(f"[*] Files longer than 5s: {over_5s}")
    print(json.dumps(stats, indent=2))

    # Long-tail aware filtering:
    # - Core set: duration <= p95
    # - Tail slice: p95 < duration <= p99, keep a fraction (deterministic subsample)
    # - Drop: duration > p99
    durations_arr = np.array(durations)
    p_core = float(np.percentile(durations_arr, args.core_percentile)) if len(durations_arr) else 0.0
    p_tail = float(np.percentile(durations_arr, args.tail_percentile)) if len(durations_arr) else 0.0

    tail_candidates = [idx for idx, d in enumerate(durations) if p_core < d <= p_tail]
    tail_keep = int(len(tail_candidates) * args.keep_tail_frac)
    if tail_candidates and tail_keep <= 0:
        tail_keep = 1  # keep at least one tail sample if tail exists
    tail_keep = min(tail_keep, len(tail_candidates))
    tail_indices = set()
    if tail_keep > 0:
        # Deterministic spread across the tail candidates
        positions = np.linspace(0, len(tail_candidates) - 1, tail_keep, dtype=int)
        tail_indices = {tail_candidates[pos] for pos in positions.tolist()}

    kept = []
    dropped = []
    invalid = []
    for idx, (wp, dur) in enumerate(zip(wav_files, durations)):
        keep_reason = None
        if dur <= p_core:
            keep_reason = "core"
        elif idx in tail_indices:
            keep_reason = "tail"
        else:
            dropped.append((wp, dur, "long_tail_drop"))
            continue

        # Basic integrity check: ensure audio is finite and non-all-zero
        try:
            wav, _ = torchaudio.load(str(wp))
            if not torch.isfinite(wav).all() or float(wav.abs().sum()) == 0.0:
                invalid.append((wp, dur, "invalid_audio"))
                continue
        except Exception:
            invalid.append((wp, dur, "load_error"))
            continue

        kept.append((wp, dur, keep_reason))
        shutil.copy(wp, out_wav_dir / wp.name)
        npz_path = npz_dir / f"{wp.stem}.npz"
        if npz_path.exists():
            shutil.copy(npz_path, out_npz_dir / npz_path.name)

    print(f"[+] Core (<=p{int(args.core_percentile)}) kept: {sum(r == 'core' for _, _, r in kept)}")
    print(
        f"[+] Tail kept (p{int(args.core_percentile)}-p{int(args.tail_percentile)} slice): "
        f"{sum(r == 'tail' for _, _, r in kept)} / {len(tail_candidates)} candidates"
    )
    print(f"[+] Dropped as long tail: {len(dropped)}")
    print(f"[!] Invalid/failed to load: {len(invalid)} (skipped)")
    print(f"[+] Copied wav to {out_wav_dir} and npz to {out_npz_dir}")

    if args.save_csv:
        import csv

        csv_path = out_dir / "durations.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_id", "duration_sec", "sample_rate", "status"])
            kept_set = {wp for wp, _, _ in kept}
            invalid_set = {wp for wp, _, _ in invalid}
            for idx, (wp, dur, sr) in enumerate(zip(wav_files, durations, sample_rates)):
                if wp in kept_set:
                    status = "kept_core" if dur <= p_core else "kept_tail"
                elif wp in invalid_set:
                    status = "invalid"
                else:
                    status = "dropped_long_tail"
                writer.writerow([wp.stem, f"{dur:.6f}", sr, status])
        print(f"[+] Saved per-file durations to {csv_path}")

    stats_path = out_dir / "duration_stats.json"
    with stats_path.open("w") as f:
        json.dump(
            {
                "stats": stats,
                "sample_rates": list(sorted(sr_set)),
                "core_percentile": args.core_percentile,
                "tail_percentile": args.tail_percentile,
                "p_core_sec": p_core,
                "p_tail_sec": p_tail,
                "tail_candidates": len(tail_candidates),
                "tail_kept": sum(r == "tail" for _, _, r in kept),
                "core_kept": sum(r == "core" for _, _, r in kept),
                "dropped_long_tail": len(dropped),
                "invalid": len(invalid),
            },
            f,
            indent=2,
        )
    print(f"[+] Saved stats to {stats_path}")

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=50, color="steelblue", edgecolor="black")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Histogram of Audio Durations (all samples)")
    plt.tight_layout()
    plot_path_all = out_dir / "duration_hist_all.png"
    plt.savefig(plot_path_all, dpi=200)
    plt.close()
    print(f"[+] Saved duration histogram (all samples) to {plot_path_all}")

    kept_durations = [dur for _, dur, _ in kept]
    if kept_durations:
        plt.figure(figsize=(8, 5))
        plt.hist(kept_durations, bins=50, color="seagreen", edgecolor="black")
        plt.xlabel("Duration (s)")
        plt.ylabel("Count")
        plt.title("Histogram of Audio Durations (kept after preprocessing)")
        plt.tight_layout()
        plot_path_kept = out_dir / "duration_hist_kept.png"
        plt.savefig(plot_path_kept, dpi=200)
        plt.close()
        print(f"[+] Saved duration histogram (kept) to {plot_path_kept}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect length distribution of sampled audio and export stats."
    )
    parser.add_argument(
        "--wav-dir",
        type=str,
        default=DEFAULT_WAV_DIR,
        help="Directory containing sampled wav files.",
    )
    parser.add_argument(
        "--npz-dir",
        type=str,
        default=DEFAULT_NPZ_DIR,
        help="Directory containing sampled npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory to store stats/results.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of wav files to analyze (0 = all).",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-file durations to CSV.",
    )
    parser.add_argument(
        "--keep-tail-frac",
        type=float,
        default=0.1,
        help="Fraction of p95-p99 tail durations to retain (deterministic subsample).",
    )
    parser.add_argument(
        "--core-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for core set (e.g., 90 for p90).",
    )
    parser.add_argument(
        "--tail-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for tail upper bound (e.g., 95 for p95).",
    )

    args = parser.parse_args()
    main(args)
