import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

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
    """Load audio with torchaudio, mono + resample."""
    waveform, sr = torchaudio.load(wav_path)
    orig_sr = sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = ta_resample(waveform, sr, target_sr)
    return waveform.squeeze(0), orig_sr


def _process_batch(
    batch_waveforms: List[torch.Tensor],
    feature_extractor: Wav2Vec2FeatureExtractor,
    wavlm_model: WavLMModel,
    target_sr: int,
    device: torch.device,
    layer_indices: List[int],
) -> torch.Tensor:
    audios = [w.cpu().numpy() for w in batch_waveforms]
    inputs = feature_extractor(
        audios, sampling_rate=target_sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wavlm_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: len = num_layers + 1 (emb + layers)

    # Stack selected layers, mean over layers, then mean-pool over time with attention_mask
    selected = torch.stack([hidden_states[i] for i in layer_indices], dim=0)  # L x B x T x D
    layer_mean = selected.mean(dim=0)  # B x T x D

    # Resize attention_mask (raw audio length) to hidden length for correct pooling
    attn_mask = inputs["attention_mask"].unsqueeze(1).float()  # B x 1 x raw_T
    attn_mask = F.interpolate(attn_mask, size=layer_mean.shape[1], mode="nearest")
    attn_mask = attn_mask.squeeze(1).unsqueeze(-1)  # B x hidden_T x 1

    masked = layer_mean * attn_mask
    lengths = attn_mask.sum(dim=1).clamp(min=1)  # B x 1
    pooled = masked.sum(dim=1) / lengths  # B x D
    return pooled.cpu()


def run_embedding(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_model)
    wavlm_model = WavLMModel.from_pretrained(args.wavlm_model).to(device).eval()
    target_sr = feature_extractor.sampling_rate
    layer_indices = list(range(args.layer_start, args.layer_end + 1))

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
    print(f"[*] WavLM model: {args.wavlm_model}")
    print(f"[*] Target sampling rate (from processor): {target_sr}")
    print(f"[*] Using layers: {layer_indices}")
    print(f"[*] Total wav files to process: {len(wav_files)}")

    for wav_path in tqdm(wav_files, desc="Extracting WavLM embeddings"):
        waveform, orig_sr = _load_waveform(wav_path, target_sr)
        if waveform is None:
            continue
        duration = waveform.shape[-1] / target_sr

        batch_waveforms.append(waveform)
        batch_info.append((wav_path, duration, orig_sr))

        if len(batch_waveforms) == args.batch_size:
            feats = _process_batch(
                batch_waveforms,
                feature_extractor,
                wavlm_model,
                target_sr,
                device,
                layer_indices,
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
            batch_waveforms,
            feature_extractor,
            wavlm_model,
            target_sr,
            device,
            layer_indices,
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
        description="Extract WavLM embeddings (mean-pooled hidden states) for sampled Bark waveforms."
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
        help="Directory to store embeddings.npy and metadata.csv.",
    )
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default="microsoft/wavlm-large",
        help="WavLM checkpoint to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for WavLM inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run WavLM on, e.g., cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of wav files to process.",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=6,
        help="First hidden state layer to include (inclusive).",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=12,
        help="Last hidden state layer to include (inclusive).",
    )

    run_embedding(parser.parse_args())

