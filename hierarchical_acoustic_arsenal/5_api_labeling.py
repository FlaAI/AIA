import argparse
import base64
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_CLUSTER_DIR = ""
DEFAULT_OUT_DIR = ""

PROMPT_SYSTEM = """
You are an expert Audio Annotator for a speech synthesis dataset. Given one audio file, output 12 scores (0-9) exactly as defined below.

Group A: Persona
- 1) Gender: 0 absolute male -> 9 absolute female
- 2) Age: 0 child -> 9 elderly
- 3) Pitch (F0, individual voice trait, emotion-independent): 0 low-pitched / bass -> 9 high-pitched / sharp
- 4) Standardization: 0 slurred / accented pronunciation -> 9 clear / standard pronunciation

Group B: Delivery
- 5) Valence: 0 totally negative (angry/sad/fear) -> 9 totally positive (happy/excited)
- 6) Prosody: 0 flat / monotone / no-pause -> 9 instable / erratic / stuttered
- 7) Energy: 0 near-silence -> 2 whisper -> 9 shout / loud voice
- 8) Speed: 0 slow -> 9 rapid

Group C: Signal
- 9) Noise Level: 0 low (studio clean) -> 9 high (strong noise)
- 10) Noise Complexity: 0 low (monotone/regular/white noise) -> 9 high (complex/irregular/music)
- 11) Spectral Texture: 0 muffled / dark / rumble -> 9 harsh / bright / metallic / screech
- 12) Glitch / Artifacts: 0 man-like / natural -> 9 robotic / heavy synthetic glitch / artifacts

Instructions:
- Listen to the audio and rate each dimension from 0 to 9 according to the scales above. Values must be integers 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 (no decimals).
- Return ONLY a single JSON object (no markdown/code fences, no extra text) with keys: {"gender": int, "age": int, "pitch": int, "standardization": int, "valence": int, "prosody": int, "energy": int, "speed": int, "noise_level": int, "noise_complexity": int, "spectral_texture": int, "glitch_artifacts": int}.
- Do not add explanations or formatting; respond with the JSON object only.
"""


def load_representatives(cluster_dir: Path, out_dir: Path, n_reps: int) -> List[Tuple[int, int, str, Path]]:
    """
    Returns list of (cluster_l1, cluster_l2, file_id, wav_path) using representatives.txt if present,
    otherwise first n_reps per child cluster. If representatives.txt exists, all listed reps are used
    (no truncation).
    """
    reps_path = out_dir / "representatives_pruned.txt"
    if not reps_path.exists():
        reps_path = cluster_dir / "representatives.txt"

    df_path = out_dir / "clustering_results_pruned.csv"
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
        for fid in candidates:
            row = group[group["file_id"] == fid].iloc[0]
            records.append((int(c1), int(c2), fid, Path(row["wav_path"])))
    return records


def encode_audio(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def call_api(
    client: OpenAI,
    model: str,
    wav_b64: str,
    force_json: bool,
) -> Dict[str, float]:
    msg = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                {"type": "text", "text": "Rate the audio per the instructions."},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"} if force_json else None,
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(f"Unexpected SDK response (no choices): {resp}")
    content = choices[0].message.content
    if content is None:
        raise RuntimeError(f"Empty response content for model {model}")
    # Try parsing JSON directly; if fenced markdown is returned, strip fences.
    parsed = None
    text = content.strip()
    if text.startswith("```"):
        # Remove leading/trailing code fences
        lines = text.splitlines()
        # drop first line (```json) and possible trailing ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        pass
    # Fallback: try to extract the first JSON object substring
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            parsed = json.loads(snippet)
            return parsed
    except Exception:
        pass
    raise RuntimeError(f"Failed to parse JSON from model response: {content}")


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.api_base or None)
    cluster_dir = Path(args.cluster_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = load_representatives(cluster_dir, out_dir, args.n_reps)
    # Apply slicing by start/end index (inclusive start, exclusive end)
    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx >= 0 else len(reps)
    end_idx = min(end_idx, len(reps))
    reps = reps[start_idx:end_idx]

    out_path = out_dir / "api_labels.json"
    # Load existing labels; reuse completed items and keep them safe.
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            results = []
    else:
        results = []

    existing_by_id = {item.get("file_id"): item for item in results if isinstance(item, dict)}

    def atomic_write(data: List[Dict]):
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)

    def prune_bad_entry(fid: str):
        """
        Create/update pruned copies of representatives and clustering_results in out_dir
        to drop problematic file_id. Originals remain untouched.
        """
        src_reps = cluster_dir / "representatives.txt"
        dst_reps = out_dir / "representatives_pruned.txt"
        if src_reps.exists() and not dst_reps.exists():
            shutil.copy(src_reps, dst_reps)
        if dst_reps.exists():
            lines = [ln for ln in dst_reps.read_text().splitlines() if fid not in ln]
            dst_reps.write_text("\n".join(lines), encoding="utf-8")

        src_df = cluster_dir / "clustering_results.csv"
        dst_df = out_dir / "clustering_results_pruned.csv"
        if src_df.exists() and not dst_df.exists():
            shutil.copy(src_df, dst_df)
        if dst_df.exists():
            try:
                df = pd.read_csv(dst_df)
                df = df[df["file_id"] != fid]
                df.to_csv(dst_df, index=False)
            except Exception:
                pass

    merged_results: List[Dict] = []
    for c1, c2, fid, wav_path in tqdm(reps, desc="API labeling", unit="wav"):
        if fid in existing_by_id:
            merged_results.append(existing_by_id[fid])
            continue
        wav_b64 = encode_audio(wav_path)
        try:
            scores = call_api(client, args.model, wav_b64, args.force_json)
            record = {
                "cluster_l1": c1,
                "cluster_l2": c2,
                "file_id": fid,
                "wav_path": str(wav_path),
                "scores": scores,
            }
            merged_results.append(record)
            existing_by_id[fid] = record
            print({"cluster_l1": c1, "cluster_l2": c2, "file_id": fid, "scores": scores})
        except Exception as e:
            print(f"[!] Failed to label {fid}: {e}")
            prune_bad_entry(fid)
        finally:
            processed_ids = {r.get("file_id") for r in merged_results if isinstance(r, dict)}
            tail_existing = [existing_by_id[k] for k in existing_by_id if k not in processed_ids]
            atomic_write(merged_results + tail_existing)

    processed_ids = {r.get("file_id") for r in merged_results if isinstance(r, dict)}
    tail_existing = [existing_by_id[k] for k in existing_by_id if k not in processed_ids]
    final_results = merged_results + tail_existing
    atomic_write(final_results)

    print(f"[+] Saved API labels for {len(final_results)} samples to {out_path}")
    for item in final_results[:5]:
        print(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call commercial API to rate 12-dim scores for representative wavs.")
    parser.add_argument("--cluster-dir", type=str, default=DEFAULT_CLUSTER_DIR, help="Cluster directory with clustering_results.csv and representatives.txt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to store api_labels.json")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the chat completion provider")
    parser.add_argument("--api-base", type=str, default="", help="Optional base URL for the API (e.g., https://api.openai-next.com/v1)")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-audio-preview",
        help="Audio-capable model name (must support input_audio blocks), e.g., gpt-4o-audio-preview",
    )
    parser.add_argument("--n-reps", type=int, default=3, help="Number of representatives per child cluster if reps file missing")
    parser.add_argument(
        "--force-json",
        action="store_true",
        help="Pass response_format={'type':'json_object'} to the API (only if the model supports it).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index (inclusive) over representatives.txt order.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End index (exclusive) over representatives.txt order; -1 means process to the end.",
    )
    args = parser.parse_args()
    main(args)

