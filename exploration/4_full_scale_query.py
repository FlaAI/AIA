import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import fcntl
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from qwen_omni_utils import process_mm_info

from transformers import utils as hf_utils
import transformers.utils.import_utils as import_utils

# Bypass torch.load safety check for local trusted checkpoints (torch<2.6)
hf_utils.check_torch_load_is_safe = lambda *a, **k: None  # noqa: E731
import_utils.check_torch_load_is_safe = lambda *a, **k: None  # noqa: E731
# Also patch within Qwen2.5 modeling module
try:
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as qwen_modeling

    qwen_modeling.check_torch_load_is_safe = lambda *a, **k: None  # noqa: E731
except Exception:
    pass


DEFAULT_UNIQUE_JSON = ""
DEFAULT_WAV_ORIG_DIR = ""
DEFAULT_WAV_TTS_DIR = ""
DEFAULT_JB_JSON = ""
DEFAULT_OUT_DIR = ""
DEFAULT_MODEL = ""

PROMPT_BUCKETS = {
    "weak": "",
    "medium": "",
    "strong": "",
}

AUDIO_TYPES = ["original", "tts_jb0", "tts_jb1", "tts_jb2"]


def load_unique_ids(path: Path) -> List[str]:
    return json.loads(path.read_text())


def load_prompts(path: Path, prompt_id: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if prompt_id:
        return [item for item in data if item.get("id") == prompt_id]
    return data


def get_audio_path(fid: str, audio_type: str, wav_orig_dir: Path, wav_tts_dir: Path) -> Path:
    if audio_type == "original":
        return wav_orig_dir / f"{fid}.wav"
    else:
        idx = int(audio_type.split("jb")[-1])
        return wav_tts_dir / f"{fid}_jb{idx:02d}.wav"


def append_with_lock(meta_path: Path, record: Dict) -> bool:
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
        key = (record.get("file_id"), record.get("audio_type"), record.get("prompt_id"))
        existing_keys = {(r.get("file_id"), r.get("audio_type"), r.get("prompt_id")) for r in existing if isinstance(r, dict)}
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


def load_existing(meta_path: Path) -> List[Dict]:
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return []


def generate_response(processor, model, wav_path: Path, text: str, device: str, dtype: torch.dtype, text_only: bool = False) -> str:
    if text_only:
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(wav_path)},
                    {"type": "text", "text": text},
                ],
            },
        ]
    chat_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=chat_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(device)
    safe_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            safe_inputs[k] = v.to(dtype)
        else:
            safe_inputs[k] = v
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        text_ids = model.generate(
            **safe_inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
        )
    gen_only = text_ids[:, prompt_len:]
    decoded = processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded.strip()


def parse_max_memory(arg: str):
    if not arg:
        return None
    chunks = [c.strip() for c in arg.split(",") if c.strip()]
    return {idx: chunk for idx, chunk in enumerate(chunks)}


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def main():
    parser = argparse.ArgumentParser(description="Full-scale audio+text queries to Qwen2.5-Omni.")
    parser.add_argument("--unique-json", type=str, default=DEFAULT_UNIQUE_JSON, help="Path to representatives_unique.json")
    parser.add_argument("--audio-type", type=str, required=True, choices=AUDIO_TYPES, help="Audio source type")
    parser.add_argument("--prompt-id", type=str, default="", help="Specific prompt id from JBB_artifacts_qwen_fine.json; empty means all.")
    parser.add_argument("--wav-orig-dir", type=str, default=DEFAULT_WAV_ORIG_DIR, help="Directory for original wav files")
    parser.add_argument("--wav-tts-dir", type=str, default=DEFAULT_WAV_TTS_DIR, help="Directory for TTS wav files")
    parser.add_argument("--jb-json", type=str, default=DEFAULT_JB_JSON, help="Path to JBB_artifacts_subset.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for results.json")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive) over unique ids")
    parser.add_argument("--end-idx", type=int, default=-1, help="End index (exclusive) over unique ids")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Qwen2.5 Omni model path or HF id")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model")
    parser.add_argument("--device-map", type=str, default="none", help="Device map ('auto' to shard, 'none' single device)")
    parser.add_argument("--text-only", action="store_true", help="If set, ignore audio and run all jailbreak texts as pure text.")
    parser.add_argument("--max-memory", type=str, default="", help="Optional comma-separated max memory per visible GPU, e.g., '7GiB,20GiB'")
    args = parser.parse_args()

    unique_ids = load_unique_ids(Path(args.unique_json)) if not args.text_only else ["text_only"]
    prompts = load_prompts(Path(args.jb_json), args.prompt_id)

    start = max(0, args.start_idx)
    end = len(unique_ids) if args.end_idx < 0 else min(len(unique_ids), args.end_idx)
    work_ids = unique_ids[start:end]

    out_dir = Path(args.output_dir)
    # Determine bucket subdir based on prompt-id membership or jb_json path
    bucket_subdir = None
    jb_json_path = Path(args.jb_json)
    for tag in ["weak", "medium", "strong"]:
        if tag in jb_json_path.parts:
            bucket_subdir = tag
            break
    if args.prompt_id:
        for bucket, ppath in PROMPT_BUCKETS.items():
            pfile = Path(ppath)
            if pfile.exists():
                try:
                    pdata = json.loads(pfile.read_text())
                    if any(str(item.get("id")) == str(args.prompt_id) for item in pdata if isinstance(item, dict)):
                        bucket_subdir = bucket
                        break
                except Exception:
                    continue
    if bucket_subdir:
        out_dir = out_dir / bucket_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_audio = "text_only" if args.text_only else args.audio_type
    meta_info: Dict[str, Dict[str, Any]] = {}
    if args.prompt_id:
        eff_prompt = args.prompt_id
        meta_path = out_dir / f"results_{safe_name(eff_audio)}_{safe_name(eff_prompt)}.json"
        existing = load_existing(meta_path)
        done_keys = {(r.get("file_id"), r.get("audio_type"), r.get("prompt_id")) for r in existing if isinstance(r, dict)}
        meta_info[eff_prompt] = {"path": meta_path, "done": done_keys}
    else:
        # prepare per-prompt output files
        for p in prompts:
            pid = str(p.get("id", ""))
            eff_prompt = pid if pid else "all"
            meta_path = out_dir / f"results_{safe_name(eff_audio)}_{safe_name(eff_prompt)}.json"
            existing = load_existing(meta_path)
            done_keys = {(r.get("file_id"), r.get("audio_type"), r.get("prompt_id")) for r in existing if isinstance(r, dict)}
            meta_info[pid] = {"path": meta_path, "done": done_keys}
        done_keys = set()  # unused in this branch

    print(f"[*] Unique ids total: {len(unique_ids)}, processing {len(work_ids)} (start={start}, end={end})")
    if args.prompt_id:
        print(f"[*] Existing records: {len(meta_info[args.prompt_id]['done'])}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.device_map == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        max_memory = parse_max_memory(args.max_memory)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
            dtype=dtype,
            max_memory=max_memory,
        ).eval()
    else:
        dtype = torch.float16 if (torch.cuda.is_available() and "cuda" in args.device) else torch.float32
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
        ).to(args.device).to(dtype).eval()

    wav_orig_dir = Path(args.wav_orig_dir)
    wav_tts_dir = Path(args.wav_tts_dir)

    for fid in tqdm(work_ids, desc="Full-scale querying", unit="fid"):
        for p in prompts:
            pid = p.get("id")
            text = p.get("prompt", "")
            # Select meta/done per prompt
            meta_path = meta_info[pid]["path"]
            done_keys_local = meta_info[pid]["done"]
            key = (fid, args.audio_type if not args.text_only else "text_only", pid)
            if key in done_keys_local:
                continue
            wav_path = None
            if not args.text_only:
                wav_path = get_audio_path(fid, args.audio_type, wav_orig_dir, wav_tts_dir)
                if not wav_path.exists():
                    print(f"[!] Missing audio {wav_path} for {fid}")
                    continue
            try:
                resp = generate_response(
                    processor,
                    model,
                    wav_path if wav_path else Path(""),
                    text,
                    args.device if args.device_map != "auto" else model.device,
                    dtype,
                    text_only=args.text_only,
                )
                record = {
                    "file_id": fid,
                    "audio_type": args.audio_type if not args.text_only else "text_only",
                    "audio_path": "" if args.text_only else str(wav_path),
                    "prompt_id": pid,
                    "prompt_text": text,
                    "response_text": resp,
                }
                appended = append_with_lock(meta_path, record)
                if appended:
                    done_keys_local.add(key)
            except Exception as e:
                print(f"[!] Error for {fid} {pid}: {repr(e)}")
                continue

    print(f"[+] Done. Results saved under {out_dir}")


if __name__ == "__main__":
    main()

