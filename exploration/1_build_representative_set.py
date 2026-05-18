import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_INPUT = ""
DEFAULT_OUT_DIR = ""


def load_data(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def select_even(entries: List[Dict], k: int) -> List[Dict]:
    """Evenly pick k items from a sorted list (by z-score)."""
    if len(entries) <= k:
        return entries
    idxs = np.linspace(0, len(entries) - 1, k, dtype=int)
    chosen = []
    for i in idxs:
        chosen.append(entries[int(i)])
    return chosen


def build_reps(data: List[Dict], k_per_bucket: int = 10) -> Dict[str, Dict[int, List[Dict]]]:
    if not data:
        return {}
    dims = list(data[0]["bucket_0_9"].keys())
    selections: Dict[str, Dict[int, List[Dict]]] = {}
    for dim in dims:
        selections[dim] = {}
        # sort by z-score for reproducibility
        sorted_entries = sorted(
            [{"file_id": d["file_id"], "score_z": float(d["scores_z"][dim]), "bucket": int(d["bucket_0_9"][dim])}
             for d in data],
            key=lambda x: x["score_z"],
        )
        # group by bucket
        buckets: Dict[int, List[Dict]] = {b: [] for b in range(10)}
        for entry in sorted_entries:
            buckets[entry["bucket"]].append(entry)
        for b in range(10):
            selections[dim][b] = select_even(buckets[b], k_per_bucket)
    return selections


def main():
    parser = argparse.ArgumentParser(description="Select per-dimension, per-bucket representatives.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to combined_scores_rank_bucket.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--k-per-bucket", type=int, default=5, help="Number of reps per bucket per dimension")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)
    selections = build_reps(data, k_per_bucket=args.k_per_bucket)

    # Union of all selected samples
    unique_ids = set()
    for dim_dict in selections.values():
        for bucket_list in dim_dict.values():
            for entry in bucket_list:
                unique_ids.add(entry["file_id"])

    (out_dir / "representatives_per_dim_bucket.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "representatives_unique.json").write_text(
        json.dumps(sorted(unique_ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Per-dim buckets saved to {out_dir / 'representatives_per_dim_bucket.json'}")
    print(f"Unique selected file_ids ({len(unique_ids)} total) saved to {out_dir / 'representatives_unique.json'}")


if __name__ == "__main__":
    main()
