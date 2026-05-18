import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


import torchaudio
torchaudio.set_audio_backend("soundfile")
from torchaudio.functional import resample as ta_resample
HAS_TORCHAUDIO = True


DEFAULT_INPUT_ROOT = ""
DEFAULT_OUTPUT_ROOT = ""
DEFAULT_WAV_DIR = os.path.join(DEFAULT_INPUT_ROOT, "wav")
DEFAULT_NPZ_DIR = os.path.join(DEFAULT_INPUT_ROOT, "npz")


def _load_waveform(
    wav_path: Path, target_sr: int
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Load audio with torchaudio only. Raise clear error if backend missing."""
    if not HAS_TORCHAUDIO or torchaudio is None or ta_resample is None:
        raise RuntimeError(
            f"torchaudio is not available. Import error: {TORCHAUDIO_IMPORT_ERR}"
        )

    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        raise RuntimeError(
            f"torchaudio.load failed for {wav_path}. "
            "Ensure torchaudio is installed with backend support (sox_io) "
            "and system libs like libsox are available."
        ) from e

    orig_sr = sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = ta_resample(waveform, sr, target_sr)
    return waveform.squeeze(0), orig_sr


def _process_batch(
    batch_waveforms: List[torch.Tensor],
    clap_processor: ClapProcessor,
    clap_model: ClapModel,
    target_sr: int,
    device: torch.device,
) -> torch.Tensor:
    audios = [w.cpu().numpy() for w in batch_waveforms]
    inputs = clap_processor(
        audios=audios, sampling_rate=target_sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = clap_model.get_audio_features(**inputs)
    return feats.cpu()


def run_embedding(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clap_processor = ClapProcessor.from_pretrained(args.clap_model)
    clap_model = ClapModel.from_pretrained(args.clap_model).to(device).eval()
    target_sr = clap_processor.feature_extractor.sampling_rate

    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    if args.max_files is not None:
        wav_files = wav_files[: args.max_files]

    if len(wav_files) == 0:
        print(f"[!] No wav files found in {wav_dir}")
        return

    embeddings = []
    metadata_rows = []
    batch_waveforms: List[torch.Tensor] = []
    batch_info: List[Tuple[Path, float, int]] = []

    print(f"[*] Using device: {device}")
    print(f"[*] CLAP model: {args.clap_model}")
    print(f"[*] Target sampling rate (from processor): {target_sr}")
    print(f"[*] Total wav files to process: {len(wav_files)}")

    for wav_path in tqdm(wav_files, desc="Extracting CLAP embeddings"):
        waveform, orig_sr = _load_waveform(wav_path, target_sr)
        if waveform is None:
            continue
        duration = waveform.shape[-1] / target_sr

        batch_waveforms.append(waveform)
        batch_info.append((wav_path, duration, orig_sr))

        if len(batch_waveforms) == args.batch_size:
            feats = _process_batch(
                batch_waveforms, clap_processor, clap_model, target_sr, device
            )
            embeddings.append(feats)
            for (fpath, dur, orig_sr), emb in zip(batch_info, feats):
                metadata_rows.append(
                    {
                        "file_id": fpath.stem,
                        "wav_path": str(fpath),
                        "npz_path": str(Path(args.npz_dir) / f"{fpath.stem}.npz"),
                        "duration_sec": dur,
                        "orig_sr": orig_sr,
                        "target_sr": target_sr,
                    }
                )
            batch_waveforms, batch_info = [], []

    if batch_waveforms:
        feats = _process_batch(
            batch_waveforms, clap_processor, clap_model, target_sr, device
        )
        embeddings.append(feats)
        for (fpath, dur, orig_sr), emb in zip(batch_info, feats):
            metadata_rows.append(
                {
                    "file_id": fpath.stem,
                    "wav_path": str(fpath),
                    "npz_path": str(Path(args.npz_dir) / f"{fpath.stem}.npz"),
                    "duration_sec": dur,
                    "orig_sr": orig_sr,
                    "target_sr": target_sr,
                }
            )

    if len(embeddings) == 0:
        print("[!] No embeddings generated.")
        return

    all_embeddings = torch.cat(embeddings, dim=0).numpy()
    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.csv"
    np.save(emb_path, all_embeddings)
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

    print(f"[+] Saved embeddings to {emb_path}")
    print(f"[+] Saved metadata to {meta_path}")
    print(f"[+] Total embeddings: {all_embeddings.shape[0]} | Dim: {all_embeddings.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLAP embeddings for sampled Bark waveforms."
    )
    parser.add_argument(
        "--wav-dir",
        type=str,
        default=DEFAULT_WAV_DIR,
        help="Directory containing wav files from sampling.",
    )
    parser.add_argument(
        "--npz-dir",
        type=str,
        default=DEFAULT_NPZ_DIR,
        help="Directory containing npz history prompts (for metadata linkage).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store embeddings.npy and metadata.parquet.",
    )
    parser.add_argument(
        "--clap-model",
        type=str,
        default="laion/clap-htsat-unfused",
        help="CLAP checkpoint to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for CLAP inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run CLAP on, e.g., cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of wav files to process.",
    )

    run_embedding(parser.parse_args())


