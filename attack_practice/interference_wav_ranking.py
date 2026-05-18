import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Defaults
DEFAULT_SUCCESS_ROOT = ""
DEFAULT_OUTPUT_DIR = ""

# Weights: dataset bucket and TTS variant
BUCKET_WEIGHTS = {"weak": 3.0, "medium": 2.0, "strong": 1.0}
TTS_WEIGHTS = {"tts_jb0": 3.0, "tts_jb1": 2.0, "tts_jb2": 1.0}


def build_file_plan(success_root: Path) -> List[Tuple[str, str, float, Path]]:
    """Return (bucket, tts_tag, weight, path) for the nine required files."""
    plan: List[Tuple[str, str, float, Path]] = []
    for bucket, b_weight in BUCKET_WEIGHTS.items():
        for tts_tag, t_weight in TTS_WEIGHTS.items():
            weight = b_weight * t_weight
            fname = f"success_rate_{bucket}_all_{tts_tag}.json"
            plan.append((bucket, tts_tag, weight, success_root / fname))
    return plan


def load_success_file(path: Path) -> Dict[str, Dict[str, float]]:
    """Load one success_rate_*.json into file_id -> metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required success file: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}")
    out: Dict[str, Dict[str, float]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        fid = item.get("file_id")
        if fid is None:
            continue
        try:
            sr = float(item.get("success_rate", 0.0))
            sc = float(item.get("avg_score", 0.0))
        except Exception:
            continue
        out[str(fid)] = {"success_rate": sr, "avg_score": sc}
    return out


def aggregate_scores(success_root: Path) -> Dict[str, Dict[str, float]]:
    """Aggregate weighted success_rate and avg_score across nine files."""
    plan = build_file_plan(success_root)
    agg: Dict[str, Dict[str, float]] = {}

    for bucket, tts_tag, weight, path in plan:
        records = load_success_file(path)
        for fid, metrics in records.items():
            entry = agg.setdefault(
                fid,
                {
                    "weighted_success": 0.0,
                    "weighted_score": 0.0,
                    "sources": [],
                },
            )
            ws = metrics["success_rate"] * weight
            wsc = metrics["avg_score"] * weight
            entry["weighted_success"] += ws
            entry["weighted_score"] += wsc
            entry["sources"].append(
                {
                    "bucket": bucket,
                    "tts_tag": tts_tag,
                    "weight": weight,
                    "success_rate": metrics["success_rate"],
                    "avg_score": metrics["avg_score"],
                    "weighted_success": ws,
                    "weighted_score": wsc,
                }
            )
    return agg


def min_max_normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = list(values.values())
    v_min, v_max = min(vals), max(vals)
    if v_max == v_min:
        return {k: 0.0 for k in values}
    return {k: (v - v_min) / (v_max - v_min) for k, v in values.items()}


def build_ranking(agg: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    weighted_success = {fid: v["weighted_success"] for fid, v in agg.items()}
    weighted_score = {fid: v["weighted_score"] for fid, v in agg.items()}

    norm_success = min_max_normalize(weighted_success)
    norm_score = min_max_normalize(weighted_score)

    ranking: List[Dict[str, float]] = []
    for fid, metrics in agg.items():
        fs = norm_success.get(fid, 0.0)
        fsc = norm_score.get(fid, 0.0)
        final = 0.5 * fs + 0.5 * fsc
        ranking.append(
            {
                "file_id": fid,
                "weighted_success": metrics["weighted_success"],
                "weighted_score": metrics["weighted_score"],
                "norm_success": fs,
                "norm_score": fsc,
                "final_score": final,
                "sources": metrics["sources"],
            }
        )

    ranking.sort(key=lambda x: x["final_score"], reverse=True)
    return ranking


def save_results(ranking: List[Dict[str, float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "ranking_full.json"

    full_payload = {
        "meta": {
            "total_candidates": len(ranking),
        },
        "items": ranking,
    }
    full_path.write_text(json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ranking_full.json for acoustic interference wav selection."
    )
    parser.add_argument(
        "--success-root",
        type=str,
        default=DEFAULT_SUCCESS_ROOT,
        help="Directory containing the nine success_rate_*_all_tts_jb*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store ranking_full.json.",
    )
    args = parser.parse_args()

    success_root = Path(args.success_root)
    output_dir = Path(args.output_dir)

    agg = aggregate_scores(success_root)
    ranking = build_ranking(agg)
    full_path = save_results(ranking, output_dir)

    print(f"[+] Aggregated {len(ranking)} candidates.")
    print(f"[+] Full ranking -> {full_path}")


if __name__ == "__main__":
    main()
