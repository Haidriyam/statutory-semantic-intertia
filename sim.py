"""
sim.py -- synthetic diachronic corpora with known graded semantic change,
PPMI-SVD embeddings, Procrustes-aligned displacement D, neighbour Jaccard J,
frequency-band noise calibration, and helper statistics.

Everything is deterministic given the seeds passed in.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import rankdata, spearmanr

# ---- generator constants -------------------------------------------------
NF = 60            # function words
NC = 1140          # content words
V = NF + NC        # vocabulary size
K = 24             # topics
SLEN = 12          # tokens per sentence
PFUNC = 0.30       # probability a token is a function word
DIM = 50           # embedding dimension
SMOOTH = 0.75      # context-distribution smoothing for PPMI
MINCOUNT = 30      # minimum count for a word to be a valid target
N_SENT = 60_000    # sentences per model (720,000 tokens)


class Language:
    """Static structure shared by all periods and arms of one 'world'."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        topic = np.tile(np.arange(K), NC // K + 1)[:NC]
        rng.shuffle(topic)
        self.topic1 = topic
        # within-topic Zipf weights (rank assigned at random inside topic)
        z = np.zeros(NC)
        for k in range(K):
            idx = np.where(topic == k)[0]
            rng.shuffle(idx)
            z[idx] = 1.0 / np.arange(1, len(idx) + 1)
        self.z = z
        pf = 1.0 / np.arange(1, NF + 1)
        self.pf = pf / pf.sum()
        self.theta = rng.dirichlet(2.0 * np.ones(K))

    def change_spec(self, p, rng):
        """Choose a random changed set and graded amounts g ~ U(0.3, 1)."""
        n_ch = int(round(p * NC))
        ch = rng.choice(NC, n_ch, replace=False)
        g = np.zeros(NC)
        g[ch] = rng.uniform(0.3, 1.0, n_ch)
        topic2 = self.topic1.copy()
        for w in ch:
            new = rng.integers(K - 1)
            topic2[w] = new if new < self.topic1[w] else new + 1
        return g, topic2

    def topic_matrix(self, g=None, topic2=None):
        """K x NC matrix of P(word | topic); period 1 if g is None."""
        M = np.zeros((K, NC))
        M[self.topic1, np.arange(NC)] += self.z if g is None else self.z * (1 - g)
        if g is not None:
            M[topic2, np.arange(NC)] += self.z * g
        return M / M.sum(1, keepdims=True)


def sample_corpus(lang, M, n_sent, rng):
    topics = rng.choice(K, n_sent, p=lang.theta)
    isf = rng.random((n_sent, SLEN)) < PFUNC
    tok = np.empty((n_sent, SLEN), dtype=np.int32)
    tok[isf] = rng.choice(NF, int(isf.sum()), p=lang.pf)
    cdf = np.cumsum(M, axis=1)
    tt = np.broadcast_to(topics[:, None], tok.shape)
    for k in range(K):
        m = (~isf) & (tt == k)
        idx = np.searchsorted(cdf[k], rng.random(int(m.sum())))
        tok[m] = NF + np.minimum(idx, NC - 1)
    return tok


class Model:
    def __init__(self, W, freq):
        self.W = W
        self.freq = freq


def embed(tok, rng):
    """Sentence-level co-occurrence -> PPMI (smoothed) -> rank-50 SVD."""
    n = tok.shape[0]
    rows = np.repeat(np.arange(n), tok.shape[1])
    S = sp.coo_matrix((np.ones(tok.size), (rows, tok.ravel())),
                      shape=(n, V)).tocsr()
    C = (S.T @ S).toarray()
    freq = np.bincount(tok.ravel(), minlength=V)
    C[np.diag_indices(V)] -= freq              # remove self-pairs
    tot = C.sum()
    pw = C.sum(1) / tot
    cs = C.sum(0) ** SMOOTH
    pc = cs / cs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C / tot / (pw[:, None] * pc[None, :]))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    v0 = rng.standard_normal(V)
    U, s, _ = svds(ppmi, k=DIM, v0=v0)
    W = U * np.sqrt(s)
    nrm = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.maximum(nrm, 1e-12)
    R, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))   # arbitrary basis
    return Model(W @ R, freq)


def build_model(lang, M, n_sent, seed):
    rng = np.random.default_rng(seed)
    return embed(sample_corpus(lang, M, n_sent, rng), rng)


def build_model_from_tokens(tok, seed):
    rng = np.random.default_rng(seed)
    return embed(tok, rng)


# ---- measures --------------------------------------------------------------
def procrustes(W1, W2, anchors):
    U, _, Vt = np.linalg.svd(W2[anchors].T @ W1[anchors])
    return U @ Vt


def displacement(m1, m2, anchors):
    Q = procrustes(m1.W, m2.W, anchors)
    return 1.0 - np.einsum("ij,ij->i", m1.W, m2.W @ Q)


def _nbr_mask(W, targets, cand, k):
    sims = W[targets] @ W[cand].T
    self_hit = targets[:, None] == cand[None, :]
    sims[self_hit] = -np.inf
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    mask = np.zeros(sims.shape, dtype=bool)
    mask[np.arange(len(targets))[:, None], top] = True
    return mask


def jaccard(m1, m2, targets, cand, k):
    a = _nbr_mask(m1.W, targets, cand, k)
    b = _nbr_mask(m2.W, targets, cand, k)
    inter = (a & b).sum(1)
    return inter / (a.sum(1) + b.sum(1) - inter)


def top_words(models, n, restrict=None):
    """n most frequent words (summed over models) with count>=MINCOUNT in all."""
    ok = np.all([m.freq >= MINCOUNT for m in models], axis=0)
    tot = np.sum([m.freq for m in models], axis=0).astype(float)
    tot[~ok] = -1
    if restrict is not None:
        mask = np.full(V, -1.0)
        mask[restrict] = tot[restrict]
        tot = mask
    order = np.argsort(-tot)
    order = order[tot[order] >= 0]
    return order[:n]


def valid_targets(models):
    ok = np.all([m.freq >= MINCOUNT for m in models], axis=0)
    ok[:NF] = False                      # content words only
    return np.where(ok)[0]


# ---- calibration -----------------------------------------------------------
def deciles(freq_t):
    """Decile index 0 (rarest) .. 9 (most frequent) over the given words."""
    r = rankdata(freq_t, method="ordinal")
    return np.minimum(((r - 1) * 10 // len(freq_t)).astype(int), 9)


def band_mean(null_vals, band):
    return np.array([null_vals[band == b].mean() if np.any(band == b) else np.nan
                     for b in range(10)])


def band_quantile(null_vals, band, q):
    return np.array([np.quantile(null_vals[band == b], q) if np.any(band == b)
                     else np.nan for b in range(10)])


# ---- statistics ------------------------------------------------------------
def auc(score, label):
    """AUC for 'higher score => label==1' (Mann-Whitney)."""
    r = rankdata(score)
    n1 = label.sum()
    n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def icc1(X):
    """ICC(1,1); X is (subjects x raters)."""
    n, k = X.shape
    gm = X.mean()
    msb = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def mean_pairwise_spearman(X):
    """X is (words x draws). Returns (mean, min) of pairwise Spearman."""
    R = np.apply_along_axis(rankdata, 0, X)
    C = np.corrcoef(R.T)
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu].mean(), C[iu].min()
