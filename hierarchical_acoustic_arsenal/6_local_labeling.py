import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import amplitude_to_DB
from tqdm import tqdm
try:
    from scipy.stats import kurtosis as sp_kurtosis
except Exception:
    sp_kurtosis = None

# Use pruned clustering outputs from API labeling stage if available
DEFAULT_CLUSTER_DIR = ""
DEFAULT_OUT_DIR = ""


def load_representatives(cluster_dir: Path, n_reps: int) -> List[Tuple[int, int, str, Path]]:
    reps_path = cluster_dir / "representatives_pruned.txt"
    if not reps_path.exists():
        reps_path = cluster_dir / "representatives.txt"

    df_path = cluster_dir / "clustering_results_pruned.csv"
    if not df_path.exists():
        df_path = cluster_dir / "clustering_results.csv"

    df = pd.read_csv(df_path)
    rep_map = {}
    if reps_path.exists():
        for line in reps_path.read_text().splitlines():
            if ":" not in line or "-" not in line:
                continue
            cluster_part, file_id = line.split(":", 1)
            try:
                l1, l2 = cluster_part.split("-")
                rep_map.setdefault((int(l1), int(l2)), []).append(file_id.strip())
            except Exception:
                continue
    records = []
    grouped = df.groupby(["cluster_l1", "cluster_l2"])
    for (c1, c2), group in grouped:
        candidates = rep_map.get((int(c1), int(c2)), [])
        if not candidates:
            candidates = group.head(n_reps)["file_id"].tolist()
        candidates = candidates[:n_reps]
        for fid in candidates:
            row = group[group["file_id"] == fid].iloc[0]
            records.append((int(c1), int(c2), fid, Path(row["wav_path"])))
    return records


def load_mono(path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def robust_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def map_to_scale(val: float, lo: float, hi: float) -> float:
    """Map to 0-9 range with clipping."""
    if hi == lo:
        return 4.5
    x = (val - lo) / (hi - lo)
    return float(np.clip(x, 0.0, 1.0) * 9.0)


def compute_features(wav: torch.Tensor, sr: int = 16000) -> Dict[str, float]:
    wav_np = wav.numpy()
    # Energy
    rms = float(torch.sqrt(torch.mean(wav ** 2)) + 1e-9)
    energy_db = float(amplitude_to_DB(torch.tensor([rms]), multiplier=20.0, amin=1e-10, db_multiplier=0.0).item())
    # Pitch (F0) using detect_pitch_frequency if available
    f0_mean = 0.0
    if hasattr(torchaudio.functional, "detect_pitch_frequency"):
        f0_series = torchaudio.functional.detect_pitch_frequency(
            wav.unsqueeze(0), sample_rate=sr, frame_time=0.032, freq_low=50, freq_high=600
        )
        f0_vals = f0_series.squeeze().numpy()
        f0_vals = f0_vals[(f0_vals > 50) & (f0_vals < 600)]
        if f0_vals.size > 0:
            f0_mean = float(np.median(f0_vals))
    # Spectral stats
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, return_complex=True)
    mag = spec.abs().numpy()
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    mag_sum = mag.sum(axis=1) + 1e-12
    centroid = float((freqs * mag_sum).sum() / mag_sum.sum())
    flatness = float(np.exp(np.mean(np.log(mag_sum))) / (np.mean(mag_sum) + 1e-12))
    # Zero-crossing for prosody/speed proxy
    zcr = float(((wav_np[:-1] * wav_np[1:]) < 0).mean())
    # Simple VAD: energy threshold
    frame_len = int(0.032 * sr)
    hop = int(0.016 * sr)
    frames = wav.unfold(0, frame_len, hop)
    frame_energy = frames.pow(2).mean(dim=1).numpy()
    vad_thresh = np.percentile(frame_energy, 60)
    speech_mask = frame_energy > vad_thresh
    speech_ratio = float(speech_mask.mean())
    # Noise level: simple SNR estimate using lower-percentile frames as noise floor
    noise_floor = float(np.percentile(frame_energy, 20))
    speech_power = float(frame_energy[speech_mask].mean()) if speech_mask.any() else float(frame_energy.mean())
    snr = float(10.0 * np.log10((speech_power + 1e-9) / (noise_floor + 1e-9)))
    # Noise complexity: spectral entropy
    psd = mag_sum / np.sum(mag_sum)
    psd = np.clip(psd, 1e-12, 1.0)
    spec_entropy = float(-np.sum(psd * np.log(psd)) / np.log(len(psd)))
    # Glitch proxy: high frequency energy ratio
    hf_mask = freqs > (sr * 0.25)
    hf_energy = float(mag_sum[hf_mask].sum())
    total_energy = float(mag_sum.sum())
    hf_ratio = hf_energy / (total_energy + 1e-9)
    # Glitch proxy: kurtosis (impulsive artifacts)
    if sp_kurtosis is not None:
        kurt = float(sp_kurtosis(wav_np, fisher=False, bias=False))
    else:
        mean = float(np.mean(wav_np))
        var = float(np.var(wav_np))
        if var > 0:
            kurt = float(np.mean((wav_np - mean) ** 4) / (var ** 2 + 1e-12))
        else:
            kurt = 0.0

    feats = {
        "energy_db": energy_db,
        "f0_mean": f0_mean,
        "centroid": centroid,
        "flatness": flatness,
        "zcr": zcr,
        "speech_ratio": speech_ratio,
        "snr_db": snr,
        "spec_entropy": spec_entropy,
        "hf_ratio": hf_ratio,
        "kurtosis": kurt,
    }
    return feats


def compute_feature_ranges(all_feats: List[Dict[str, float]], q_lo: float, q_hi: float) -> Dict[str, Tuple[float, float]]:
    """Compute per-feature robust ranges using percentiles over all samples."""
    ranges: Dict[str, Tuple[float, float]] = {}
    if not all_feats:
        return ranges
    keys = all_feats[0].keys()
    for k in keys:
        vals = [float(f.get(k, 0.0)) for f in all_feats if np.isfinite(f.get(k, 0.0))]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        lo = float(np.percentile(arr, q_lo))
        hi = float(np.percentile(arr, q_hi))
        if hi <= lo:
            hi = lo + 1e-6
        ranges[k] = (lo, hi)
    return ranges


def score_from_features(feats: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Heuristic mapping to 0-9 using dataset-driven percentile ranges."""

    def rget(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        return ranges.get(name, default)

    energy = map_to_scale(feats["energy_db"], *rget("energy_db", (-50.0, -10.0)))
    pitch = map_to_scale(feats["f0_mean"], *rget("f0_mean", (80.0, 320.0)))  # Hz
    speed = map_to_scale(feats["speech_ratio"], *rget("speech_ratio", (0.2, 0.9)))
    prosody = map_to_scale(feats["zcr"], *rget("zcr", (0.02, 0.15)))
    noise_level = map_to_scale(feats["flatness"], *rget("flatness", (0.1, 0.9)))
    noise_complexity = map_to_scale(feats["spec_entropy"], *rget("spec_entropy", (0.3, 0.95)))
    spectral_texture = map_to_scale(feats["centroid"], *rget("centroid", (500.0, 5000.0)))
    glitch_hf = map_to_scale(feats["hf_ratio"], *rget("hf_ratio", (0.05, 0.3)))
    glitch_kurt = map_to_scale(feats["kurtosis"], *rget("kurtosis", (1.0, 10.0)))
    glitch_artifacts = (glitch_hf + glitch_kurt) / 2.0

    # Rough estimates for subjective axes using proxies
    gender = map_to_scale(feats["f0_mean"], *rget("f0_mean", (80.0, 250.0)))
    # For age proxy, invert f0_mean range
    age_lo, age_hi = rget("f0_mean", (250.0, 80.0))
    age = map_to_scale(feats["f0_mean"], age_hi, age_lo)
    valence = 4.5  # cannot infer reliably in this heuristic; midpoint of 0-9
    standardization = map_to_scale(feats["speech_ratio"], *rget("speech_ratio", (0.2, 0.9)))
    authenticity = map_to_scale(1.0 - feats["flatness"], 1.0 - rget("flatness", (0.1, 0.9))[1], 1.0 - rget("flatness", (0.1, 0.9))[0])

    return {
        "gender": round(gender, 2),
        "age": round(age, 2),
        "pitch": round(pitch, 2),
        "standardization": round(standardization, 2),
        "valence": round(valence, 2),
        "prosody": round(prosody, 2),
        "energy": round(energy, 2),
        "speed": round(speed, 2),
        "noise_level": round(noise_level, 2),
        "noise_complexity": round(noise_complexity, 2),
        "spectral_texture": round(spectral_texture, 2),
        "glitch_artifacts": round(glitch_artifacts, 2),
        "authenticity": round(authenticity, 2),
    }


def main(args):
    cluster_dir = Path(args.cluster_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = load_representatives(cluster_dir, args.n_reps)
    items = []
    for c1, c2, fid, wav_path in tqdm(reps, desc="Local refining (pass 1: features)", unit="wav"):
        wav = load_mono(wav_path, target_sr=args.target_sr)
        feats = compute_features(wav, sr=args.target_sr)
        items.append(
            {
                "cluster_l1": c1,
                "cluster_l2": c2,
                "file_id": fid,
                "wav_path": str(wav_path),
                "features": feats,
            }
        )

    ranges = compute_feature_ranges([it["features"] for it in items], args.range_lower, args.range_upper)

    results = []
    for it in tqdm(items, desc="Local refining (pass 2: scoring)", unit="wav"):
        scores = score_from_features(it["features"], ranges)
        it_out = {
            **it,
            "scores": scores,
        }
        results.append(it_out)

    with open(out_dir / "local_scores.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved local scores for {len(results)} samples to {out_dir / 'local_scores.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local DSP-based scoring for 12-dim attributes (heuristic).")
    parser.add_argument("--cluster-dir", type=str, default=DEFAULT_CLUSTER_DIR, help="Cluster directory with clustering_results.csv and representatives.txt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to store local_scores.json")
    parser.add_argument("--n-reps", type=int, default=3, help="Number of representatives per child cluster if reps file missing")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sampling rate for analysis")
    parser.add_argument("--range-lower", type=float, default=0.0, help="Lower percentile for feature ranges (e.g., 2)")
    parser.add_argument("--range-upper", type=float, default=100.0, help="Upper percentile for feature ranges (e.g., 98)")
    args = parser.parse_args()
    main(args)
