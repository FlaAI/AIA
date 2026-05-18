import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

try:
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import matplotlib
import matplotlib.colors as mcolors


DEFAULT_CLAP_EMB_ROOT = ""
DEFAULT_WAVLM_EMB_ROOT = ""
DEFAULT_OUT_ROOT = ""


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def kmeans_cluster(x: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 1 or len(x) == 0:
        labels = np.zeros(len(x), dtype=int)
        centroids = np.mean(x, axis=0, keepdims=True) if len(x) > 0 else np.zeros((1, x.shape[1]))
        return labels, centroids
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(x)
    return labels, km.cluster_centers_


def diversity_filter(
    embs: np.ndarray,
    indices: np.ndarray,
    centroid: np.ndarray,
    threshold: float,
    rep_k: int,
) -> Tuple[Optional[int], List[int]]:
    """
    One baseline (closest to centroid) + up to rep_k farthest diverse outliers.
    """
    if len(indices) == 0:
        return None, []
    sub = embs[indices]
    dists = 1 - np.dot(sub, centroid)
    base_local = int(np.argmin(dists))
    base_idx = int(indices[base_local])

    order = np.argsort(-dists)
    selected: List[int] = []
    for idx in order:
        gidx = int(indices[idx])
        if gidx == base_idx:
            continue
        if not selected:
            selected.append(gidx)
        else:
            sims = np.dot(embs[selected], sub[idx])
            if np.all(sims <= threshold):
                selected.append(gidx)
        if len(selected) >= rep_k:
            break
    return base_idx, selected


def maybe_umap(embs: np.ndarray, seed: int) -> Optional[np.ndarray]:
    if not HAS_UMAP:
        print("[!] umap-learn not installed. Skipping UMAP.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(embs)


def maybe_tsne(embs: np.ndarray, seed: int) -> Optional[np.ndarray]:
    reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    return reducer.fit_transform(embs)


def maybe_pca(embs: np.ndarray) -> Optional[np.ndarray]:
    if embs.shape[1] < 2:
        return None
    x = embs - embs.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        comps = vt[:2]
        proj = np.dot(x, comps.T)
        return proj
    except Exception as e:
        print(f"[!] PCA failed: {e}")
        return None


def get_discrete_cmap(k: int) -> mcolors.ListedColormap:
    """
    Build a discrete colormap with at least k distinguishable colors.
    Combines tab20 / tab20b / tab20c (60 colors) and falls back to hsv if k>60.
    """
    tab_palettes = []
    for name in ["tab20", "tab20b", "tab20c"]:
        try:
            tab_palettes.extend(matplotlib.colormaps[name].colors)
        except Exception:
            tab_palettes.extend(plt.get_cmap(name).colors)  # type: ignore
    if k <= len(tab_palettes):
        colors = tab_palettes[:k]
    else:
        colors = matplotlib.colormaps["hsv"](np.linspace(0, 1, k))
    return mcolors.ListedColormap(colors)


def load_and_align(clap_dir: Path, wavlm_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    clap_emb = np.load(clap_dir / "embeddings.npy")
    clap_meta = pd.read_csv(clap_dir / "metadata.csv")
    wavlm_emb = np.load(wavlm_dir / "embeddings.npy")
    wavlm_meta = pd.read_csv(wavlm_dir / "metadata.csv")

    # Align by file_id intersection
    clap_meta["__idx_clap"] = range(len(clap_meta))
    wavlm_meta["__idx_wavlm"] = range(len(wavlm_meta))
    merged = clap_meta[["file_id", "__idx_clap"]].merge(
        wavlm_meta[["file_id", "__idx_wavlm"]], on="file_id", how="inner"
    )
    if len(merged) == 0:
        raise ValueError("No overlapping file_id between CLAP and WavLM metadata.")
    if len(merged) < max(len(clap_meta), len(wavlm_meta)):
        print(f"[!] Warning: aligned {len(merged)} samples; CLAP {len(clap_meta)}, WavLM {len(wavlm_meta)}.")

    clap_emb = clap_emb[merged["__idx_clap"].to_numpy()]
    wavlm_emb = wavlm_emb[merged["__idx_wavlm"].to_numpy()]
    metadata = clap_meta.loc[merged["__idx_clap"]].drop(columns="__idx_clap").reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    return clap_emb, wavlm_emb, metadata


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clap_dir = Path(args.clap_embedding_dir)
    wavlm_dir = Path(args.wavlm_embedding_dir)
    clap_emb, wavlm_emb, metadata = load_and_align(clap_dir, wavlm_dir)

    clap_norm = l2_normalize(clap_emb)
    wavlm_norm = l2_normalize(wavlm_emb)
    embs_combo = np.concatenate([wavlm_norm, clap_norm], axis=1)
    embs_combo = l2_normalize(embs_combo)

    l1_labels, l1_centroids = kmeans_cluster(embs_combo, args.k_top, args.seed)

    l2_labels = np.zeros(len(embs_combo), dtype=int)
    keep_flags = np.zeros(len(embs_combo), dtype=bool)
    rep_tags: List[str] = []

    stats: Dict[str, Dict] = {"level1": {}, "level2": {}}

    for c1 in range(args.k_top):
        idx_l1 = np.where(l1_labels == c1)[0]
        if len(idx_l1) == 0:
            continue
        sub_embs = embs_combo[idx_l1]
        # Adapt child k to cluster size if requested
        if args.adaptive_child:
            target_k = int(len(idx_l1) / args.adaptive_divisor)
            k_child = max(args.adaptive_min, min(target_k, args.adaptive_max))
        else:
            k_child = args.k_child
        k_child = min(k_child, len(idx_l1))
        child_labels, child_centroids = kmeans_cluster(sub_embs, k_child, args.seed)

        for child in range(k_child):
            idx_sub = idx_l1[np.where(child_labels == child)[0]]
            if len(idx_sub) == 0:
                continue
            centroid = child_centroids[child]
            base_idx, outliers = diversity_filter(
                embs_combo, idx_sub, centroid, args.sim_threshold, args.rep_per_cluster
            )
            l2_labels[idx_sub] = child
            keep_idx = set()
            if base_idx is not None:
                keep_idx.add(base_idx)
                rep_tags.append(f"{c1}-{child}:{metadata.loc[base_idx, 'file_id']}")
            for r in outliers:
                keep_idx.add(r)
                rep_tags.append(f"{c1}-{child}:{metadata.loc[r, 'file_id']}")
            if keep_idx:
                keep_flags[list(keep_idx)] = True

    df = metadata.copy()
    df["cluster_l1"] = l1_labels
    df["cluster_l2"] = l2_labels
    df["keep"] = keep_flags

    # Optional projections and plots
    cmap = get_discrete_cmap(args.k_top)
    if args.enable_pca:
        coords = maybe_pca(embs_combo)
        if coords is not None:
            df["pca_x"] = coords[:, 0]
            df["pca_y"] = coords[:, 1]
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=df["cluster_l1"],
                    cmap=cmap,
                    s=6,
                    alpha=0.7,
                )
                cb = plt.colorbar(scatter)
                cb.set_ticks([])
                cb.set_label("first-level clusters")
                plt.xlabel("PCA dimension 1")
                plt.ylabel("PCA dimension 2")
                plt.title("PCA Projection of WavLM + CLAP Embeddings")
                plt.tight_layout()
                plt.savefig(out_dir / "pca_plot.png", dpi=200)
                plt.close()
            except Exception as e:
                print(f"[!] Failed to save PCA plot: {e}")

    if args.enable_tsne:
        try:
            coords = maybe_tsne(embs_combo, args.seed)
            df["tsne_x"] = coords[:, 0]
            df["tsne_y"] = coords[:, 1]
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=df["cluster_l1"],
                    cmap=cmap,
                    s=6,
                    alpha=0.7,
                )
                cb = plt.colorbar(scatter)
                cb.set_ticks([])
                cb.set_label("first-level clusters")
                plt.xlabel("t-SNE dimension 1")
                plt.ylabel("t-SNE dimension 2")
                plt.title("t-SNE Projection of WavLM + CLAP Embeddings")
                plt.tight_layout()
                plt.savefig(out_dir / "tsne_plot.png", dpi=200)
                plt.close()
            except Exception as e:
                print(f"[!] Failed to save t-SNE plot: {e}")
        except Exception as e:
            print(f"[!] t-SNE failed: {e}")

    if args.enable_umap:
        coords = maybe_umap(embs_combo, args.seed)
        if coords is not None:
            df["umap_x"] = coords[:, 0]
            df["umap_y"] = coords[:, 1]
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=df["cluster_l1"],
                    cmap=cmap,
                    s=6,
                    alpha=0.7,
                )
                cb = plt.colorbar(scatter)
                cb.set_ticks([])
                cb.set_label("first-level clusters")
                plt.xlabel("UMAP dimension 1")
                plt.ylabel("UMAP dimension 2")
                plt.title("UMAP Projection of WavLM + CLAP Embeddings")
                plt.tight_layout()
                plt.savefig(out_dir / "umap_plot.png", dpi=200)
                plt.close()
            except Exception as e:
                print(f"[!] Failed to save UMAP plot: {e}")

    stats["level1"]["k"] = args.k_top
    stats["level1"]["counts"] = {int(k): int(v) for k, v in pd.Series(l1_labels).value_counts().items()}
    stats["level2"]["k_child"] = args.k_child
    stats["level2"]["adaptive_child"] = bool(args.adaptive_child)
    stats["level2"]["adaptive_divisor"] = args.adaptive_divisor
    stats["level2"]["adaptive_min"] = args.adaptive_min
    stats["level2"]["adaptive_max"] = args.adaptive_max
    stats["level2"]["sim_threshold"] = args.sim_threshold
    stats["level2"]["rep_per_cluster"] = args.rep_per_cluster
    stats["kept_total"] = int(keep_flags.sum())
    stats["total"] = int(len(df))

    df.to_csv(out_dir / "clustering_results.csv", index=False)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(out_dir / "representatives.txt", "w") as f:
        for tag in rep_tags:
            f.write(tag + "\n")

    # Print top-level cluster counts (total and kept representatives)
    count_l1 = pd.Series(l1_labels).value_counts().sort_index()
    keep_l1 = pd.Series(l1_labels[keep_flags]).value_counts().sort_index()
    print("[*] Top-level cluster sizes (total | kept):")
    for cid, total in count_l1.items():
        kept = int(keep_l1.get(cid, 0))
        print(f"    L1 {int(cid):02d}: total={int(total)}, kept={kept}")

    print(f"[+] Saved clustering to {out_dir / 'clustering_results.csv'}")
    print(f"[+] Stats: kept {stats['kept_total']} / {stats['total']}")
    print(f"[+] Representatives saved to {out_dir / 'representatives.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering with diversity filtering on WavLM + CLAP embeddings (sklearn k-means)."
    )
    parser.add_argument(
        "--clap-embedding-dir",
        type=str,
        default=DEFAULT_CLAP_EMB_ROOT,
        help="Directory containing CLAP embeddings.npy and metadata.csv.",
    )
    parser.add_argument(
        "--wavlm-embedding-dir",
        type=str,
        default=DEFAULT_WAVLM_EMB_ROOT,
        help="Directory containing WavLM embeddings.npy and metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUT_ROOT,
        help="Directory to store clustering results.",
    )
    parser.add_argument(
        "--k-top",
        type=int,
        default=20,
        help="Top-level K for k-means.",
    )
    parser.add_argument(
        "--k-child",
        type=int,
        default=10,
        help="Child-level K for each parent cluster.",
    )
    parser.add_argument(
        "--adaptive-child",
        action="store_true",
        help="If set, derive child K per cluster size: k = clamp(int(size/divisor), adaptive_min, adaptive_max).",
    )
    parser.add_argument(
        "--adaptive-divisor",
        type=float,
        default=20.0,
        help="Divisor used when adaptive-child is enabled (k ~ size/divisor).",
    )
    parser.add_argument(
        "--adaptive-min",
        type=int,
        default=5,
        help="Minimum child K when adaptive-child is enabled.",
    )
    parser.add_argument(
        "--adaptive-max",
        type=int,
        default=20,
        help="Maximum child K when adaptive-child is enabled.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for diversity filtering.",
    )
    parser.add_argument(
        "--rep-per-cluster",
        type=int,
        default=3,
        help="Number of representative samples to record per child cluster.",
    )
    parser.add_argument(
        "--enable-umap",
        action="store_true",
        help="Enable UMAP projection to 2D (umap-learn required).",
    )
    parser.add_argument(
        "--enable-tsne",
        action="store_true",
        help="Enable t-SNE projection to 2D.",
    )
    parser.add_argument(
        "--enable-pca",
        action="store_true",
        help="Enable PCA projection to 2D and save plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    run(parser.parse_args())

