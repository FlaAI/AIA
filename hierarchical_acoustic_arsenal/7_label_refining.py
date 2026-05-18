import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


WEIGHTS = {
    "gender": 0.8, # 0.5
    "age": 0.8,
    "pitch": 0.2,
    "standardization": 0.5,
    "valence": 1.0,  # no local info
    "prosody": 0.8,
    "energy": 0.2,
    "speed": 0.2,
    "noise_level": 0.2,
    "noise_complexity": 0.5,
    "spectral_texture": 0.2,
    "glitch_artifacts": 0.2
}


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text())
    out = {}
    for item in data:
        fid = item.get("file_id")
        sc = item.get("scores", {})
        if fid is None or not sc:
            continue
        out[fid] = {k: float(v) for k, v in sc.items()}
    return out


def summarize(scores: List[Dict[str, float]], hist_bins: List[int], hist_name: str) -> Dict[str, Dict]:
    if not scores:
        return {}
    dims = list(scores[0].keys())
    summary = {}
    for dim in dims:
        vals = [float(s.get(dim, 0.0)) for s in scores]
        arr = np.array(vals, dtype=float)
        cnt = Counter(int(v) for v in vals)
        summary[dim] = {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            hist_name: {str(k): int(cnt.get(k, 0)) for k in hist_bins},
        }
    return summary


def zscore(values: List[float]) -> Tuple[List[float], float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=0)) if arr.size else 0.0
    if std == 0:
        return [0.0] * len(values), mean, std
    return ((arr - mean) / std).tolist(), mean, std


def compute_zscores(api: Dict[str, Dict[str, float]], local: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    ids = sorted(set(api.keys()) & set(local.keys()))
    if not ids:
        return {}, {}
    dims = list(next(iter(api.values())).keys())

    api_z: Dict[str, Dict[str, float]] = {}
    local_z: Dict[str, Dict[str, float]] = {}
    for dim in dims:
        api_vals = [api[i][dim] for i in ids]
        local_vals = [local[i][dim] for i in ids]
        api_norm, _, _ = zscore(api_vals)
        local_norm, _, _ = zscore(local_vals)
        for fid, az, lz in zip(ids, api_norm, local_norm):
            api_z.setdefault(fid, {})[dim] = float(az)
            local_z.setdefault(fid, {})[dim] = float(lz)
    return api_z, local_z


def combine_scores(api: Dict[str, Dict[str, float]], local: Dict[str, Dict[str, float]], api_z: Dict[str, Dict[str, float]], local_z: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    combined = []
    for fid in sorted(set(api.keys()) & set(local.keys())):
        merged_raw = {}
        merged_z = {}
        for dim, a_val in api[fid].items():
            lam = WEIGHTS.get(dim, 0.5)
            merged_raw[dim] = lam * float(a_val) + (1.0 - lam) * float(local[fid].get(dim, 0.0))
            merged_z[dim] = lam * float(api_z[fid].get(dim, 0.0)) + (1.0 - lam) * float(local_z[fid].get(dim, 0.0))
        combined.append({"file_id": fid, "scores_raw": merged_raw, "scores_z": merged_z})
    return combined


def round_scores(combined: List[Dict[str, float]]) -> List[Dict[str, float]]:
    rounded = []
    for item in combined:
        fid = item["file_id"]
        raw = item["scores_raw"]
        scores = {}
        for k, v in raw.items():
            scores[k] = int(np.clip(round(v), 0, 9))
        rounded.append({"file_id": fid, "scores": scores, "scores_raw": raw})
    return rounded


def rank_bucket(combined: List[Dict[str, float]], num_bins: int = 10) -> List[Dict[str, Dict[str, float]]]:
    """
    Rank combined z-scores per dimension and assign equal-count bins [0..num_bins-1].
    """
    if not combined:
        return []
    dims = list(combined[0]["scores_z"].keys())
    buckets_per_dim: Dict[str, Dict[str, int]] = {d: {} for d in dims}
    # Build per-dim arrays
    for dim in dims:
        vals = [(item["file_id"], float(item["scores_z"][dim])) for item in combined]
        sorted_vals = sorted(vals, key=lambda x: x[1])
        n = len(sorted_vals)
        if n == 0:
            continue
        edges = [int(np.ceil((i / num_bins) * n)) for i in range(1, num_bins)]
        edges = [min(max(e, 0), n) for e in edges]
        current_bin = 0
        for idx, (fid, _) in enumerate(sorted_vals):
            while current_bin < len(edges) and idx >= edges[current_bin]:
                current_bin += 1
            buckets_per_dim[dim][fid] = current_bin

    bucketed = []
    for item in combined:
        fid = item["file_id"]
        bucket_labels = {dim: buckets_per_dim.get(dim, {}).get(fid, 0) for dim in dims}
        bucketed.append(
            {
                "file_id": fid,
                "scores_z": item["scores_z"],
                "bucket_0_9": bucket_labels,
            }
        )
    return bucketed


def main(args):
    api_dir = Path(args.api_dir)
    local_dir = Path(args.local_dir)
    api_path = api_dir / "api_labels.json"
    local_path = local_dir / "local_scores.json"
    if not api_path.exists() or not local_path.exists():
        raise FileNotFoundError("api_labels.json or local_scores.json not found.")

    api_scores = load_scores(api_path)
    local_scores = load_scores(local_path)

    api_z, local_z = compute_zscores(api_scores, local_scores)
    combined = combine_scores(api_scores, local_scores, api_z, local_z)
    rounded = round_scores(combined)  # for raw score summary only
    bucketed = rank_bucket(combined, num_bins=10)

    # Summaries (raw scores, not rank buckets)
    score_dicts_10 = [r["scores"] for r in rounded]
    summary_10 = summarize(score_dicts_10, hist_bins=list(range(10)), hist_name="hist_0_9")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Raw score outputs (as before, for actual 0-9 values and summary)
    (out_dir / "combined_scores_raw.json").write_text(json.dumps(rounded, ensure_ascii=False, indent=2))
    (out_dir / "combined_stats_raw_10.json").write_text(json.dumps(summary_10, ensure_ascii=False, indent=2))

    # Rank-based equal-count buckets (0-9) with combined z scores
    (out_dir / "combined_scores_rank_bucket.json").write_text(json.dumps(bucketed, ensure_ascii=False, indent=2))

    print(f"[+] Raw combined scores (with rounding) -> {out_dir / 'combined_scores_raw.json'}")
    print(f"[+] Rank buckets (equal count 0-9) -> {out_dir / 'combined_scores_rank_bucket.json'}")
    print(f"[+] Raw stats (0-9 summary) -> {out_dir / 'combined_stats_raw_10.json'}")

    print("\n=== 0-9 raw summary ===")
    for dim, stats in summary_10.items():
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(stats["hist_0_9"].items(), key=lambda x: int(x[0])))
        print(
            f"{dim}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}; hist [{hist}]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine API and local scores, round to 0-9, and also produce relative 1-5 bins."
    )
    parser.add_argument("--api-dir", type=str, required=True, help="Directory containing api_labels.json")
    parser.add_argument("--local-dir", type=str, required=True, help="Directory containing local_scores.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store combined outputs")
    args = parser.parse_args()
    main(args)
