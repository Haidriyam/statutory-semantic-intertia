"""
Diachronic Vector Space Alignment and Semantic Displacement Engine
Author: Muhammad Adeel Sajjad

Reference implementation of Orthogonal Procrustes alignment and nearest-neighbor
stability metrics for tracking lexical displacement across temporal text partitions.
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


class StatutoryInertiaEngine:
    def __init__(
        self, 
        vector_dim: int = 100, 
        window: int = 5, 
        min_count: int = 2, 
        seed: int = 42
    ):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count
        self.seed = seed

    def train_epoch(self, tokenized_corpus: List[List[str]]) -> Word2Vec:
        """Trains a Skip-gram Word2Vec model on a temporal corpus partition."""
        if not tokenized_corpus:
            raise ValueError("Corpus cannot be empty.")
        
        return Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,              # Skip-gram
            seed=self.seed,
            workers=1,         # Deterministic training
            epochs=10
        )

    @staticmethod
    def align_spaces(
        base_model: Word2Vec, 
        target_model: Word2Vec, 
        anchor_vocab: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Maps target_model coordinates into base_model coordinate frame
        via Orthogonal Procrustes (Schönemann, 1966) using SVD.
        """
        shared = sorted(
            list(set(base_model.wv.index_to_key).intersection(set(target_model.wv.index_to_key)))
        )
        
        # If no dedicated anchors provided, use top shared words
        if anchor_vocab:
            anchors = [w for w in anchor_vocab if w in shared]
        else:
            anchors = shared

        if len(anchors) < 5:
            raise ValueError(f"Insufficient anchor tokens for alignment: {len(anchors)} found.")

        # Extract anchor matrices
        w_base = np.array([base_model.wv[w] for w in anchors], dtype=float)
        w_target = np.array([target_model.wv[w] for w in anchors], dtype=float)

        # Row-normalize to decouple frequency magnitude from directional alignment
        w_base_norm = w_base / np.linalg.norm(w_base, axis=1, keepdims=True)
        w_target_norm = w_target / np.linalg.norm(w_target, axis=1, keepdims=True)

        # SVD of cross-covariance matrix: M = W_target.T @ W_base
        u, _, vt = np.linalg.svd(np.dot(w_target_norm.T, w_base_norm), full_matrices=False)
        q_rot = np.dot(u, vt)

        aligned_vectors = {}
        for w in target_model.wv.index_to_key:
            vec = target_model.wv[w].astype(float)
            aligned_vectors[w] = np.dot(vec, q_rot)

        return shared, aligned_vectors

    @staticmethod
    def compute_metrics(
        base_model: Word2Vec, 
        aligned_target: Dict[str, np.ndarray], 
        eval_words: List[str], 
        k_neighbors: int = 5
    ) -> List[Dict]:
        """
        Computes cosine displacement and top-k Jaccard neighborhood overlap.
        """
        shared_candidates = sorted(
            list(set(base_model.wv.index_to_key).intersection(set(aligned_target.keys())))
        )
        
        # Build normalized candidate matrices for matrix-vector neighbor lookups
        c_base = np.array([base_model.wv[w] for w in shared_candidates], dtype=float)
        c_target = np.array([aligned_target[w] for w in shared_candidates], dtype=float)

        norm_base = np.linalg.norm(c_base, axis=1, keepdims=True)
        norm_target = np.linalg.norm(c_target, axis=1, keepdims=True)

        # Guard against zero vectors
        norm_base[norm_base == 0] = 1e-12
        norm_target[norm_target == 0] = 1e-12

        c_base /= norm_base
        c_target /= norm_target

        cand_to_idx = {w: i for i, w in enumerate(shared_candidates)}
        records = []

        for word in eval_words:
            if word not in cand_to_idx:
                continue

            v1 = base_model.wv[word].astype(float)
            v2 = aligned_target[word].astype(float)

            # Cosine displacement D(w) = 1 - cos(v1, v2_aligned)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue

            cos_sim = float(np.dot(v1, v2) / (norm_v1 * norm_v2))
            cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
            disp = 1.0 - cos_sim

            # Find top-k nearest neighbors in base period (excluding self)
            sims_base = np.dot(c_base, (v1 / norm_v1))
            self_idx = cand_to_idx[word]
            sims_base[self_idx] = -np.inf
            top_k_base_idx = np.argpartition(sims_base, -k_neighbors)[-k_neighbors:]
            neighbors_base = {shared_candidates[i] for i in top_k_base_idx}

            # Find top-k nearest neighbors in aligned target period (excluding self)
            sims_target = np.dot(c_target, (v2 / norm_v2))
            sims_target[self_idx] = -np.inf
            top_k_target_idx = np.argpartition(sims_target, -k_neighbors)[-k_neighbors:]
            neighbors_target = {shared_candidates[i] for i in top_k_target_idx}

            # Jaccard overlap
            union = neighbors_base.union(neighbors_target)
            jaccard = len(neighbors_base.intersection(neighbors_target)) / float(len(union)) if union else 0.0

            # Legacy composite (exploratory; kept for benchmark tracking)
            composite_lsi = (1.0 - disp) * (0.5 + 0.5 * jaccard)

            records.append({
                "word": word,
                "displacement": disp,
                "jaccard": jaccard,
                "lsi_exploratory": composite_lsi,
                "neighbors_t1": list(neighbors_base),
                "neighbors_t2": list(neighbors_target)
            })

        # Rank by displacement ascending (lowest drift at the top)
        records.sort(key=lambda x: x["displacement"])
        return records

    def plot_displacement_spectrum(
        self, 
        records: List[Dict], 
        output_filename: str = "output/semantic_change_distribution.png"
    ):
        """Generates horizontal bar chart of relative vector displacement."""
        if not records:
            print("[-] No records to plot.")
            return

        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        
        words = [r["word"] for r in records]
        displacements = [r["displacement"] for r in records]

        plt.figure(figsize=(9, 4.5), dpi=300)
        y_pos = np.arange(len(words))
        
        # Color coding: cooler for low drift, warmer for high displacement
        colors = ["#1f4e5b" if d < 0.20 else "#b33939" for d in displacements]

        plt.barh(y_pos, displacements, color=colors, edgecolor="black", linewidth=0.6)
        plt.yticks(y_pos, words, fontsize=9)
        plt.xlabel("Cosine Displacement $D(w) = 1 - \\cos(v_{t1}, Q^T v_{t2})$", fontsize=10)
        plt.title("Diachronic Lexical Displacement (Reference Demonstration)", fontsize=11, fontweight="bold", pad=10)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"[+] Saved displacement artifact: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reference Procrustes alignment on test corpora.")
    parser.add_argument("--output", type=str, default="output/semantic_change_distribution.png", help="Path for output figure.")
    parser.add_argument("--k", type=int, default=3, help="Nearest neighbors for Jaccard calculation.")
    args = parser.parse_args()

    # Controlled baseline text slices (Synthetic illustrative draws)
    corpus_epoch1 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["a", "considerable", "hazard", "threatened", "the", "shipping", "enterprise"],
        ["the", "nice", "distinction", "required", "exact", "calculation", "and", "judgment"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 35

    corpus_epoch2 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["an", "environmental", "hazard", "created", "toxic", "waste", "contamination"],
        ["they", "had", "a", "nice", "pleasant", "dinner", "conversation"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 35

    print("[*] Training period models (Skip-gram, d=50)...")
    engine = StatutoryInertiaEngine(vector_dim=50, window=4, min_count=2, seed=42)
    m1 = engine.train_epoch(corpus_epoch1)
    m2 = engine.train_epoch(corpus_epoch2)

    print("[*] Aligning vector spaces via SVD Procrustes...")
    shared_vocab, aligned_target = engine.align_spaces(m1, m2)

    # Specific target probe words
    test_probe = ["hereinabove", "indenture", "witnesseth", "statute", "hazard", "nice"]
    eval_set = [w for w in test_probe if w in shared_vocab]

    results = engine.compute_metrics(m1, aligned_target, eval_set, k_neighbors=args.k)

    print("\n" + "=" * 65)
    print(f"{'Token':<16}{'Displacement (D)':<20}{'Jaccard (J)':<14}{'Legacy LSI':<12}")
    print("=" * 65)
    for r in results:
        print(f"{r['word']:<16}{r['displacement']:<20.4f}{r['jaccard']:<14.2f}{r['lsi_exploratory']:<12.4f}")
    print("=" * 65)
    print("[!] Note: Metrics derived on synthetic demonstration data.")

    engine.plot_displacement_spectrum(results, output_filename=args.output)
