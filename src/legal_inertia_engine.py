"""
Diachronic Vector Space Alignment and Semantic Displacement Engine
Author: Muhammad Adeel Sajjad

Reference implementation of Orthogonal Procrustes alignment and nearest-neighbor
stability metrics for tracking lexical displacement across temporal text partitions.
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


class StatutoryInertiaEngine:
    def __init__(
        self, 
        vector_dim: int = 30, 
        window: int = 3, 
        min_count: int = 1, 
        seed: int = 42
    ):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count
        self.seed = seed

    def train_epoch(self, tokenized_corpus: List[List[str]]) -> Word2Vec:
        """Trains a Skip-gram Word2Vec model on a temporal corpus partition."""
        return Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,              # Skip-gram
            seed=self.seed,
            workers=1,         # Strict determinism
            epochs=50,         # Sufficient iterations to converge on small benchmark draws
            compute_loss=True
        )

    @staticmethod
    def align_spaces(
        base_model: Word2Vec, 
        target_model: Word2Vec, 
        anchor_tokens: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Maps target_model coordinates into base_model coordinate frame
        via Orthogonal Procrustes (Schonemann, 1966) using SVD.
        """
        shared = sorted(
            list(set(base_model.wv.index_to_key).intersection(set(target_model.wv.index_to_key)))
        )
        
        # Anchor selection: Use stable background tokens, excluding test probe words
        if anchor_tokens:
            anchors = [w for w in anchor_tokens if w in shared]
        else:
            anchors = shared

        if len(anchors) < 3:
            raise ValueError(f"Insufficient anchors for alignment: {len(anchors)} found.")

        w_base = np.array([base_model.wv[w] for w in anchors], dtype=float)
        w_target = np.array([target_model.wv[w] for w in anchors], dtype=float)

        # Normalize anchor vectors before computing cross-covariance
        w_base_norm = w_base / np.linalg.norm(w_base, axis=1, keepdims=True)
        w_target_norm = w_target / np.linalg.norm(w_target, axis=1, keepdims=True)

        # Orthogonal Procrustes: M = W_target^T @ W_base = U @ Sigma @ V^T -> Q = U @ V^T
        m_cov = np.dot(w_target_norm.T, w_base_norm)
        u, _, vt = np.linalg.svd(m_cov, full_matrices=False)
        q_rot = np.dot(u, vt)

        aligned_vectors = {
            w: np.dot(target_model.wv[w].astype(float), q_rot)
            for w in target_model.wv.index_to_key
        }

        return shared, aligned_vectors

    @staticmethod
    def compute_metrics(
        base_model: Word2Vec, 
        aligned_target: Dict[str, np.ndarray], 
        eval_words: List[str], 
        k_neighbors: int = 3
    ) -> List[Dict]:
        """Computes cosine displacement and top-k Jaccard neighborhood overlap."""
        shared_candidates = sorted(
            list(set(base_model.wv.index_to_key).intersection(set(aligned_target.keys())))
        )
        
        c_base = np.array([base_model.wv[w] for w in shared_candidates], dtype=float)
        c_target = np.array([aligned_target[w] for w in shared_candidates], dtype=float)

        c_base /= np.linalg.norm(c_base, axis=1, keepdims=True)
        c_target /= np.linalg.norm(c_target, axis=1, keepdims=True)

        cand_to_idx = {w: i for i, w in enumerate(shared_candidates)}
        records = []

        for word in eval_words:
            if word not in cand_to_idx:
                continue

            v1 = base_model.wv[word].astype(float)
            v2 = aligned_target[word].astype(float)

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue

            cos_sim = float(np.dot(v1, v2) / (norm_v1 * norm_v2))
            cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
            disp = 1.0 - cos_sim

            # Top-k neighbors in Epoch 1
            idx = cand_to_idx[word]
            sims_1 = np.dot(c_base, (v1 / norm_v1))
            sims_1[idx] = -np.inf
            top_1 = np.argpartition(sims_1, -k_neighbors)[-k_neighbors:]
            n_t1 = {shared_candidates[i] for i in top_1}

            # Top-k neighbors in Epoch 2 (aligned)
            sims_2 = np.dot(c_target, (v2 / norm_v2))
            sims_2[idx] = -np.inf
            top_2 = np.argpartition(sims_2, -k_neighbors)[-k_neighbors:]
            n_t2 = {shared_candidates[i] for i in top_2}

            union_n = n_t1.union(n_t2)
            jaccard = len(n_t1.intersection(n_t2)) / float(len(union_n)) if union_n else 0.0
            composite_lsi = (1.0 - disp) * (0.5 + 0.5 * jaccard)

            records.append({
                "word": word,
                "displacement": disp,
                "jaccard": jaccard,
                "lsi_exploratory": composite_lsi,
                "neighbors_t1": list(n_t1),
                "neighbors_t2": list(n_t2)
            })

        records.sort(key=lambda x: x["displacement"])
        return records

    def plot_displacement_spectrum(
        self, 
        records: List[Dict], 
        output_filename: str = "output/semantic_change_distribution.png",
        drift_threshold: float = 0.20
    ):
        """Generates horizontal bar chart with distinct color bands for stable vs. drifted tokens."""
        if not records:
            return

        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        
        words = [r["word"] for r in records]
        displacements = [r["displacement"] for r in records]

        plt.figure(figsize=(9, 4.5), dpi=300)
        y_pos = np.arange(len(words))
        
        # Color coding: stable statutory anchors in teal, drifted vernacular in crimson
        colors = ["#1f4e5b" if d < drift_threshold else "#b33939" for d in displacements]

        bars = plt.barh(y_pos, displacements, color=colors, edgecolor="black", linewidth=0.6)
        plt.yticks(y_pos, words, fontsize=10, weight="medium")
        plt.xlabel("Cosine Displacement $D(w) = 1 - \\cos(v_{t1}, Q^T v_{t2})$", fontsize=10)
        plt.title("Diachronic Lexical Displacement (Reference Demonstration)", fontsize=11, fontweight="bold", pad=12)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.35)
        
        # Add numeric labels next to each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.015, 
                bar.get_y() + bar.get_height() / 2, 
                f"{width:.3f}", 
                va="center", 
                ha="left", 
                fontsize=8.5, 
                color="#333333"
            )

        plt.xlim(0, max(displacements) * 1.15)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"[+] Successfully saved updated visualization: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reference Procrustes alignment on test corpora.")
    parser.add_argument("--output", type=str, default="output/semantic_change_distribution.png")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    # Controlled baseline text slices
    corpus_epoch1 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["a", "considerable", "hazard", "threatened", "the", "shipping", "enterprise", "at", "sea"],
        ["the", "nice", "distinction", "required", "exact", "calculation", "and", "judgment"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 60

    corpus_epoch2 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["an", "environmental", "hazard", "created", "toxic", "waste", "contamination", "in", "water"],
        ["they", "had", "a", "nice", "pleasant", "dinner", "conversation", "yesterday"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 60

    print("[*] Training period models (Skip-gram, d=30)...")
    engine = StatutoryInertiaEngine(vector_dim=30, window=3, min_count=1, seed=42)
    m1 = engine.train_epoch(corpus_epoch1)
    m2 = engine.train_epoch(corpus_epoch2)

    # Use persistent structural syntax words as alignment anchors
    anchor_tokens = ["the", "court", "decrees", "relief", "provides", "remedy", "under", "seal", "severe", "penalties"]

    print("[*] Aligning vector spaces via SVD Procrustes on structural anchors...")
    shared_vocab, aligned_target = engine.align_spaces(m1, m2, anchor_tokens=anchor_tokens)

    # Evaluate statutory terms vs. drifted vernacular terms
    test_probe = ["hereinabove", "indenture", "witnesseth", "statute", "hazard", "nice"]
    eval_set = [w for w in test_probe if w in shared_vocab]

    results = engine.compute_metrics(m1, aligned_target, eval_set, k_neighbors=args.k)

    print("\n" + "=" * 65)
    print(f"{'Token':<16}{'Displacement (D)':<20}{'Jaccard (J)':<14}{'Legacy LSI':<12}")
    print("=" * 65)
    for r in results:
        print(f"{r['word']:<16}{r['displacement']:<20.4f}{r['jaccard']:<14.2f}{r['lsi_exploratory']:<12.4f}")
    print("=" * 65)

    engine.plot_displacement_spectrum(results, output_filename=args.output)
