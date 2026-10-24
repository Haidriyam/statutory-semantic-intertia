"""
Statutory Semantic Inertia & Fossilization Engine
Aligns chronological statutory vs. vernacular vector spaces via SVD Procrustes
to quantify institutional semantic resistance and LLM interpretative misalignment.
"""

import sys
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


class StatutoryInertiaEngine:
    def __init__(self, vector_dim: int = 100, window: int = 5, min_count: int = 1):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count

    def train_epoch(self, tokenized_corpus: list) -> Word2Vec:
        """Trains Word2Vec skip-gram embeddings for a specific temporal epoch."""
        return Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,
            seed=42,
            workers=1,
        )

    @staticmethod
    def align_spaces(base_model: Word2Vec, target_model: Word2Vec):
        """
        Orthogonal Procrustes Matrix Alignment:
        Finds optimal orthogonal matrix Q maximizing trace(W_base * (W_target * Q)^T).
        """
        shared_vocab = sorted(
            list(
                set(base_model.wv.index_to_key).intersection(
                    set(target_model.wv.index_to_key)
                )
            )
        )
        if len(shared_vocab) < 5:
            raise ValueError("Insufficient overlapping vocabulary for alignment.")

        w_base = np.array([base_model.wv[w] for w in shared_vocab])
        w_target = np.array([target_model.wv[w] for w in shared_vocab])

        # Singular Value Decomposition of cross-covariance matrix
        correlation_matrix = np.dot(w_target.T, w_base)
        u, _, vt = np.linalg.svd(correlation_matrix)
        q = np.dot(u, vt)

        aligned_vectors = {
            w: np.dot(target_model.wv[w], q)
            for w in target_model.wv.index_to_key
        }
        return shared_vocab, aligned_vectors

    @staticmethod
    def compute_metrics(base_model: Word2Vec, aligned_target: dict, shared_vocab: list, k_neighbors: int = 3):
        """
        Calculates:
        1. Cosine Displacement: D(w) = 1 - cos(v_t1, v_t2)
        2. Neighborhood Jaccard Stability: Overlap of top-k semantic neighbors
        3. Legal Semantic Inertia Index (LSI): (1 - D(w)) * Jaccard_Stability
        """
        records = []
        for word in shared_vocab:
            v_t1 = base_model.wv[word]
            v_t2 = aligned_target[word]

            # Displacement
            disp = float(cosine(v_t1, v_t2))

            # Neighborhood stability (Jaccard)
            neighbors_t1 = set(
                [w for w, _ in base_model.wv.similar_by_vector(v_t1, topn=k_neighbors + 1) if w != word]
            )
            # Find nearest neighbors in aligned target space
            target_sims = []
            for target_w, target_vec in aligned_target.items():
                if target_w != word:
                    sim = 1.0 - float(cosine(v_t2, target_vec))
                    target_sims.append((target_w, sim))
            target_sims.sort(key=lambda x: x[1], reverse=True)
            neighbors_t2 = set([w for w, _ in target_sims[:k_neighbors]])

            jaccard = (
                len(neighbors_t1.intersection(neighbors_t2))
                / float(len(neighbors_t1.union(neighbors_t2)))
                if (neighbors_t1 or neighbors_t2)
                else 0.0
            )

            # High LSI = High institutional resistance to semantic drift
            lsi = (1.0 - disp) * (0.5 + 0.5 * jaccard)

            records.append({
                "word": word,
                "displacement": disp,
                "jaccard": jaccard,
                "lsi": lsi,
                "institutional_class": "Statutory Anchor (Fossilized)" if lsi > 0.70 else "Living Vernacular (Drifted)"
            })

        records.sort(key=lambda x: x["lsi"], reverse=True)
        return records

    def plot_inertia_spectrum(self, records: list, output_filename="inertia_distribution.png"):
        words = [r["word"] for r in records]
        lsi_scores = [r["lsi"] for r in records]
        colors = ["#1b4965" if s > 0.70 else "#c1121f" for s in lsi_scores]

        plt.figure(figsize=(10, 5), dpi=300)
        plt.barh(words, lsi_scores, color=colors, edgecolor="black", linewidth=0.6)
        plt.axvline(x=0.70, color="#6c757d", linestyle="--", linewidth=1.2, label="Statutory Anchor Threshold (LSI > 0.70)")
        plt.xlabel("Legal Semantic Inertia Index (LSI)", fontsize=10, weight="semibold")
        plt.title("Distribution of Semantic Inertia: Statutory Precedent vs. Vernacular Drift", fontsize=11, weight="bold", pad=12)
        plt.gca().invert_yaxis()
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"[+] Saved visualization artifact: {output_filename}")


if __name__ == "__main__":
    # Epoch 1: Historical Baseline (Early Statutory Enactments & General 19th-C. Prose)
    corpus_epoch1 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["a", "considerable", "hazard", "threatened", "the", "shipping", "enterprise"],
        ["the", "nice", "distinction", "required", "exact", "calculation", "and", "judgment"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 30

    # Epoch 2: Contemporary Era (Modern Commercial/Regulatory Contexts)
    # Notice: 'hazard' and 'nice' shift context, while statutory terms ('hereinabove', 'indenture') remain frozen
    corpus_epoch2 = [
        ["the", "statute", "provides", "remedy", "hereinabove", "specified", "in", "the", "indenture"],
        ["the", "court", "decrees", "relief", "witnesseth", "lawful", "execution", "hereunto"],
        ["an", "environmental", "hazard", "created", "toxic", "waste", "contamination"],
        ["they", "had", "a", "nice", "pleasant", "dinner", "conversation"],
        ["severe", "penalties", "attach", "to", "wilful", "default", "under", "seal"]
    ] * 30

    engine = StatutoryInertiaEngine()
    print("[*] Training historical statutory baseline model...")
    m1 = engine.train_epoch(corpus_epoch1)

    print("[*] Training contemporary comparison model...")
    m2 = engine.train_epoch(corpus_epoch2)

    print("[*] Performing Orthogonal Procrustes alignment (SVD)...")
    shared_v, aligned_t2 = engine.align_spaces(m1, m2)

    results = engine.compute_metrics(m1, aligned_t2, shared_v)

    print("\n" + "=" * 68)
    print(f"{'Token':<16}{'Displacement':<16}{'Jaccard':<12}{'LSI Index':<12}{'Status':<16}")
    print("=" * 68)
    for r in results:
        print(f"{r['word']:<16}{r['displacement']:<16.4f}{r['jaccard']:<12.2f}{r['lsi']:<12.4f}{r['institutional_class']:<16}")
    print("=" * 68)

    engine.plot_inertia_spectrum(results)
