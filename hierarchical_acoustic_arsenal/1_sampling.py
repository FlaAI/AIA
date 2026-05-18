import argparse
import os
import csv
import numpy as np
import random
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm
import torch

from bark.generation import (
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    codec_decode,
    preload_models,
)

# Config
SEED_TEXT = "The quick brown fox jumps over the lazy dog."
NUM_SAMPLES = 5
OUTPUT_ROOT = ""
SAMPLE_RATE = 24000

NPZ_DIR = os.path.join(OUTPUT_ROOT, "npz")
WAV_DIR = os.path.join(OUTPUT_ROOT, "wav")
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)


def _to_int32_np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.int32)
    return np.array(arr, dtype=np.int32)


def run_extraction(start_id: int = 0):
    """Controlled Monte Carlo voice mining for Bark latent styles."""
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

    print("[*] Starting mining process.")
    print(f"[*] Seed Text: '{SEED_TEXT}'")
    print(f"[*] Target Samples: {NUM_SAMPLES}")
    print(f"[*] Starting file index: {start_id}")

    temp_records = []

    for i in tqdm(range(NUM_SAMPLES), desc="Mining Acoustic Latents"):
        try:
            with torch.no_grad():
                # Dynamic temperatures to inject acoustic diversity
                coarse_temp = 0.7 if random.random() < 0.7 else random.uniform(0.8, 1.1)
                fine_temp = 0.5 if random.random() < 0.7 else random.uniform(0.8, 1.2)

                # history_prompt=None forces random speaker/style sampling
                semantic_tokens = generate_text_semantic(
                    SEED_TEXT,
                    history_prompt=None,
                    temp=0.7,
                    top_k=50,
                    top_p=0.95,
                )

                coarse_tokens = generate_coarse(
                    semantic_tokens,
                    history_prompt=None,
                    temp=coarse_temp,
                    top_k=50,
                    top_p=0.95,
                )

                fine_tokens = generate_fine(
                    coarse_tokens,
                    history_prompt=None,
                    temp=fine_temp,
                )

                audio_arr = codec_decode(fine_tokens)

            file_id = f"voice_{start_id + i:05d}"

            # Save Bark prompts (future history_prompt)
            np.savez(
                os.path.join(NPZ_DIR, f"{file_id}.npz"),
                semantic_prompt=_to_int32_np(semantic_tokens),
                coarse_prompt=_to_int32_np(coarse_tokens),
                fine_prompt=_to_int32_np(fine_tokens),
            )

            # Save waveform for downstream embedding extraction
            audio_np = audio_arr.cpu().numpy() if isinstance(audio_arr, torch.Tensor) else audio_arr
            audio_data = (audio_np * 32767).astype(np.int16)
            write_wav(os.path.join(WAV_DIR, f"{file_id}.wav"), SAMPLE_RATE, audio_data)

            temp_records.append((file_id, coarse_temp, fine_temp))

            if i % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Error at sample {i}: {e}")
            continue

    # Persist temperature log for downstream analysis/replay
    temp_log_path = os.path.join(OUTPUT_ROOT, "temperature_log.csv")
    with open(temp_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "coarse_temp", "fine_temp"])
        writer.writerows(temp_records)

    print(f"\n[Done] All {NUM_SAMPLES} samples saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[Warning] CUDA not available. This will be very slow. Aborting recommended.")
    else:
        parser = argparse.ArgumentParser(description="Sample Bark latents with optional start index.")
        parser.add_argument(
            "--start-id",
            type=int,
            default=0,
            help="Starting numeric id for file naming (voice_<id>).",
        )
        args = parser.parse_args()
        run_extraction(start_id=args.start_id)
