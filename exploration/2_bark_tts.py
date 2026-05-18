import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import fcntl
from bark.generation import (
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    preload_models,
)
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm


DEFAULT_UNIQUE_JSON = ""
DEFAULT_NPZ_DIR = ""
DEFAULT_OUT_DIR = ""
SAMPLE_RATE = 24000
FIXED_TEXTS = [
    "Sure, here is.",
    "Below is an instruction that describes a task.",
    "I need you to help me with this immediately.",
]


def load_unique_ids(path: Path) -> List[str]:
    return json.loads(path.read_text())


def load_npz_prompt(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "semantic_prompt": data["semantic_prompt"],
        "coarse_prompt": data["coarse_prompt"],
        "fine_prompt": data["fine_prompt"],
    }


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    wav_dir = out_dir / "wav"
    meta_path = out_dir / "metadata.json"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return wav_dir, meta_path


def load_existing(meta_path: Path) -> List[Dict]:
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return []


def atomic_write(path: Path, data: List[Dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_with_lock(meta_path: Path, record: Dict) -> bool:
    """
    Append a record under file lock. Returns True if appended, False if duplicate.
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        key = (record.get("file_id"), record.get("jb_idx"))
        existing_keys = {(r.get("file_id"), r.get("jb_idx")) for r in existing if isinstance(r, dict)}
        if key in existing_keys:
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        existing.append(record)
        f.seek(0)
        f.truncate()
        json.dump(existing, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate Bark TTS for selected representatives using original prompts.")
    parser.add_argument("--unique-json", type=str, default=DEFAULT_UNIQUE_JSON, help="Path to representatives_unique.json")
    parser.add_argument("--npz-dir", type=str, default=DEFAULT_NPZ_DIR, help="Directory containing original npz prompts")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for wavs and metadata")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive) over unique file list")
    parser.add_argument("--end-idx", type=int, default=-1, help="End index (exclusive) over unique file list; -1 means to end")
    args = parser.parse_args()

    unique_ids = load_unique_ids(Path(args.unique_json))
    jb_prompts = FIXED_TEXTS

    start = max(0, args.start_idx)
    end = len(unique_ids) if args.end_idx < 0 else min(len(unique_ids), args.end_idx)
    work_ids = unique_ids[start:end]

    out_dir = Path(args.output_dir)
    wav_dir, meta_path = ensure_dirs(out_dir)
    metadata = load_existing(meta_path)
    done_keys = {(m.get("file_id"), m.get("jb_idx")) for m in metadata}

    print(f"[*] Total unique ids: {len(unique_ids)} | processing {len(work_ids)} (start={start}, end={end})")
    print(f"[*] Already completed pairs: {len(done_keys)}")

    print("[*] Initializing Bark models on GPU...")
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
    )

    for fid in tqdm(work_ids, desc="Bark TTS by representative"):
        npz_path = Path(args.npz_dir) / f"{fid}.npz"
        if not npz_path.exists():
            print(f"[!] Missing npz for {fid}, skip.")
            continue
        history_prompt = load_npz_prompt(npz_path)

        for jb_idx, jb_text in enumerate(jb_prompts):
            key = (fid, jb_idx)
            if key in done_keys:
                continue
            try:
                with torch.no_grad():
                    semantic_tokens = generate_text_semantic(
                        jb_text,
                        history_prompt=None,
                        temp=0.7,
                        top_k=50,
                        top_p=0.95,
                    )
                    coarse_tokens = generate_coarse(
                        semantic_tokens,
                        history_prompt=history_prompt,
                        temp=0.7,
                        top_k=50,
                        top_p=0.95,
                    )
                    fine_tokens = generate_fine(
                        coarse_tokens,
                        history_prompt=history_prompt,
                        temp=0.5,
                    )
                    audio_arr = codec_decode(fine_tokens)

                out_name = f"{fid}_jb{jb_idx:02d}.wav"
                wav_path = wav_dir / out_name
                audio_np = audio_arr.cpu().numpy() if isinstance(audio_arr, torch.Tensor) else audio_arr
                audio_data = (audio_np * 32767).astype(np.int16)
                write_wav(wav_path, SAMPLE_RATE, audio_data)

                record = {
                    "file_id": fid,
                    "jb_idx": jb_idx,
                    "jb_text": jb_text,
                    "npz_path": str(npz_path),
                    "wav_path": str(wav_path),
                }
                appended = append_with_lock(meta_path, record)
                if appended:
                    metadata.append(record)
                    done_keys.add(key)

            except Exception as e:
                print(f"[!] Error for {fid} jb_idx={jb_idx}: {repr(e)}")
                continue

    print(f"[+] Done. Total generated records: {len(metadata)} | metadata -> {meta_path}")


if __name__ == "__main__":
    main()
