"""
real_pipeline.py -- run the noise-calibrated protocol on real text.

Input format everywhere: plain text (optionally .gz), ONE SENTENCE PER LINE,
already lower-cased/tokenised (whitespace-separated). SemEval-2020 lemma files
already have this form.

Subcommands
  semeval  validate D / E_D / J against the SemEval-2020 Task 1 gold graded scores
  paired   paired general-vs-statutory comparison (Delta E_D, bootstrap CI)
  prep     turn raw .txt files into one-sentence-per-line lower-cased text

Design points (see paper, Sec. IV-C): shared vocabulary and anchors across all
models; equal token budget n per model; split-half null from period 1 (needs
>= 2n tokens); optional near-duplicate removal (MinHash) before training.
"""
import argparse, glob, gzip, json, os, re, zlib
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
import sim


# ---------------------------------------------------------------- I/O ------
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.endswith(".gz") \
        else open(path, encoding="utf-8", errors="ignore")


def read_sentences(paths):
    out = []
    for p in sorted(paths):
        with _open(p) as f:
            for line in f:
                t = line.split()
                if len(t) >= 3:
                    out.append(t)
    return out


def expand(spec):
    """File, directory or glob -> list of files."""
    if os.path.isdir(spec):
        return [p for p in glob.glob(os.path.join(spec, "**", "*"), recursive=True) if os.path.isfile(p)]
    return glob.glob(spec) or ([spec] if os.path.exists(spec) else [])


def n_tokens(sents):
    return sum(len(s) for s in sents)


# ------------------------------------------------------ near-duplicates ----
def dedup(sents, min_len=8, shingle=3, n_hash=64, bands=16, thr=0.8, seed=0):
    """Drop exact and near-duplicate lines (MinHash + banding, Jaccard >= thr).
    Lines shorter than min_len tokens are kept (too short to judge)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, n_hash, dtype=np.int64)
    b = rng.integers(0, 2**31 - 1, n_hash, dtype=np.int64)
    P = 2**31 - 1
    keep, buckets, sigs = [], {}, {}
    rows = n_hash // bands
    for i, s in enumerate(sents):
        if len(s) < min_len:
            keep.append(i); continue
        sh = {zlib.crc32(" ".join(s[j:j + shingle]).encode()) for j in range(len(s) - shingle + 1)}
        x = np.fromiter(sh, dtype=np.int64)
        sig = ((a[:, None] * x[None, :] + b[:, None]) % P).min(1)
        dup = False
        for k in range(bands):
            key = (k, sig[k * rows:(k + 1) * rows].tobytes())
            for j in buckets.get(key, ()):
                if (sigs[j] == sig).mean() >= thr:
                    dup = True; break
            if dup: break
        if not dup:
            for k in range(bands):
                buckets.setdefault((k, sig[k * rows:(k + 1) * rows].tobytes()), []).append(i)
            sigs[i] = sig
            keep.append(i)
    return [sents[i] for i in keep]


# -------------------------------------------------- vocabulary / models ----
def build_vocab(corpora, size, min_count, force=()):
    c = Counter()
    for sents in corpora:
        for s in sents:
            c.update(s)
    words = [w for w, n in c.most_common() if n >= min_count][:size]
    for w in force:
        if w in c and w not in words:
            words.append(w)
    return {w: i for i, w in enumerate(words)}


def to_ids(sents, vocab):
    out = []
    for s in sents:
        ids = np.array([vocab[w] for w in s if w in vocab], dtype=np.int32)
        if len(ids) >= 2:
            out.append(ids)
    return out


def take_tokens(ids_sents, order, start, budget):
    """Consecutive sentences from a permutation `order` until budget tokens."""
    got, n, i = [], 0, start
    while i < len(order) and n < budget:
        s = ids_sents[order[i]]
        got.append(s); n += len(s); i += 1
    return got, i, n


def embed_real(ids_sents, V, dim, window, rng, smooth=0.75):
    """Windowed (within-sentence) co-occurrence -> PPMI -> truncated SVD."""
    tok = np.concatenate(ids_sents)
    sid = np.repeat(np.arange(len(ids_sents)), [len(s) for s in ids_sents])
    freq = np.bincount(tok, minlength=V)
    C = sp.csr_matrix((V, V), dtype=np.float64)
    CH = 2_000_000
    for o in range(1, window + 1):
        same = sid[:-o] == sid[o:]
        r, c = tok[:-o][same], tok[o:][same]
        for st in range(0, len(r), CH):
            rr, cc = r[st:st + CH], c[st:st + CH]
            M = sp.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(V, V)).tocsr()
            C = C + M + M.T
    C = C.tocoo()
    tot = C.data.sum()
    rs = np.asarray(C.tocsr().sum(1)).ravel()
    cs = np.asarray(C.tocsr().sum(0)).ravel() ** smooth
    pw = rs / tot
    pc = cs / cs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C.data / tot / (pw[C.row] * pc[C.col]))
    keep = np.isfinite(pmi) & (pmi > 0)
    P = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=(V, V))
    U, s, _ = svds(P, k=dim, v0=rng.standard_normal(V))
    W = U * np.sqrt(s)
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    return sim.Model(W, freq)


# ------------------------------------------------------- core scoring ------
def train_triplet(s1, s2, vocab, n, dim, window, rng):
    """Equal-budget models: period-1 (A), period-1 disjoint null (B), period-2."""
    V = len(vocab)
    i1, i2 = to_ids(s1, vocab), to_ids(s2, vocab)
    o1, o2 = rng.permutation(len(i1)), rng.permutation(len(i2))
    A, pos, na = take_tokens(i1, o1, 0, n)
    B, _, nb = take_tokens(i1, o1, pos, n)
    P2, _, n2 = take_tokens(i2, o2, 0, n)
    if min(na, nb, n2) < 0.95 * n:
        raise ValueError(f"not enough tokens for budget n={n}: got {na},{nb},{n2}")
    return [embed_real(x, V, dim, window, rng) for x in (A, P2, B)]


def score_triplet(models, targets, anchors, cand, k):
    m1, m2, mn = models
    D = sim.displacement(m1, m2, anchors)[targets]
    D0 = sim.displacement(m1, mn, anchors)[targets]
    J = sim.jaccard(m1, m2, targets, cand, k)
    J0 = sim.jaccard(m1, mn, targets, cand, k)
    band = sim.deciles(m1.freq[targets])
    ED = D - sim.band_mean(D0, band)[band]
    EJ = sim.band_mean(J0, band)[band] - J
    return {"D": D, "J": J, "ED": ED, "EJ": EJ, "D0": D0}


def pick_sets(all_models, targets_pool, n_anchor, n_cand, exclude):
    """Anchors/candidates: most frequent words common to all models, minus targets."""
    ok = np.all([m.freq >= sim.MINCOUNT for m in all_models], axis=0)
    tot = np.sum([m.freq for m in all_models], axis=0).astype(float)
    tot[~ok] = -1
    tot[list(exclude)] = -1
    order = np.argsort(-tot)
    order = order[tot[order] > 0]
    return order[:n_anchor], order[:n_cand]


# ------------------------------------------------------------ SemEval ------
def load_semeval(root):
    tg = [l.strip().split()[0] for l in open(os.path.join(root, "targets.txt")) if l.strip()]
    gold = {}
    for l in open(os.path.join(root, "truth", "graded.txt")):
        p = l.split()
        if len(p) >= 2:
            gold[p[0]] = float(p[1])
    def files(c):
        f = expand(os.path.join(root, c, "lemma")) or expand(os.path.join(root, c))
        if not f:
            raise FileNotFoundError(f"no corpus files under {root}/{c}")
        return f
    return tg, gold, files("corpus1"), files("corpus2")


def cmd_semeval(a):
    rng = np.random.default_rng(a.seed)
    tg, gold, f1, f2 = load_semeval(a.root)
    s1, s2 = read_sentences(f1), read_sentences(f2)
    if a.dedup:
        s1, s2 = dedup(s1), dedup(s2)
    vocab = build_vocab([s1, s2], a.vocab, a.min_count, force=tg)
    tg = [w for w in tg if w in vocab and w in gold]
    tid = np.array([vocab[w] for w in tg])
    n = a.budget or min(n_tokens(s1) // 2, n_tokens(s2))
    print(f"targets={len(tg)} vocab={len(vocab)} budget n={n:,} tokens", flush=True)
    res = {"n_targets": len(tg), "budget": int(n), "seeds": []}
    g = np.array([gold[w] for w in tg])
    rhos = {k: [] for k in ("D", "ED", "negJ", "EJ")}
    for r in range(a.runs):
        models = train_triplet(s1, s2, vocab, n, a.dim, a.window, rng)
        ok = np.all([m.freq[tid] >= sim.MINCOUNT for m in models], axis=0)
        print(f"Targets passing frequency filter: {int(ok.sum())}/{len(tid)}", flush=True)
        anchors, cand = pick_sets(models, tid, a.anchors, a.cand, exclude=tid)
        sc = score_triplet(models, tid, anchors, cand, a.k)
        for key, vals in (("D", sc["D"]), ("ED", sc["ED"]), ("negJ", -sc["J"]), ("EJ", sc["EJ"])):
            rhos[key].append(float(spearmanr(vals[ok], g[ok])[0]))
        print(f"run {r}: " + " ".join(f"{k}={v[-1]:.3f}" for k, v in rhos.items()), flush=True)
    res["spearman_vs_gold"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
                               for k, v in rhos.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["spearman_vs_gold"], indent=1))


# ------------------------------------------------------------- paired ------
def cmd_paired(a):
    rng = np.random.default_rng(a.seed)
    arms = {"gen": (read_sentences(expand(a.gen_p1)), read_sentences(expand(a.gen_p2))),
            "stat": (read_sentences(expand(a.stat_p1)), read_sentences(expand(a.stat_p2)))}
    if a.dedup_stat:
        arms["stat"] = tuple(dedup(s) for s in arms["stat"])      # compare with/without
    vocab = build_vocab([s for pair in arms.values() for s in pair], a.vocab, a.min_count)
    n = a.budget or min(min(n_tokens(p1) // 2, n_tokens(p2)) for p1, p2 in arms.values())
    print(f"vocab={len(vocab)} budget n={n:,} tokens per model", flush=True)
    out = {"budget": int(n), "dedup_stat": bool(a.dedup_stat), "runs": []}
    for r in range(a.runs):
        mods = {k: train_triplet(p1, p2, vocab, n, a.dim, a.window, rng) for k, (p1, p2) in arms.items()}
        allm = mods["gen"] + mods["stat"]
        ok = np.all([m.freq >= a.min_target_count for m in allm], axis=0)
        targets = np.where(ok)[0]
        anchors, cand = pick_sets(allm, targets, a.anchors, a.cand, exclude=[])
        sg = score_triplet(mods["gen"], targets, anchors, cand, a.k)
        ss = score_triplet(mods["stat"], targets, anchors, cand, a.k)
        rec = {"n_targets": int(len(targets))}
        idx = rng.integers(0, len(targets), (1000, len(targets)))
        for key in ("D", "ED", "J", "EJ"):
            d = sg[key] - ss[key]
            bs = d[idx].mean(1)
            rec[f"delta_{key}"] = {"mean": float(d.mean()), "lo": float(np.quantile(bs, .025)),
                                   "hi": float(np.quantile(bs, .975))}
        out["runs"].append(rec)
        print(f"run {r}: dED={rec['delta_ED']['mean']:+.4f} "
              f"[{rec['delta_ED']['lo']:+.4f},{rec['delta_ED']['hi']:+.4f}] "
              f"dEJ={rec['delta_EJ']['mean']:+.4f} n={len(targets)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def cmd_prep(a):
    sent_re = re.compile(r"(?<=[.;:!?])\s+")
    with open(a.out, "w") as fo:
        for p in expand(a.inp):
            with _open(p) as f:
                for para in f.read().split("\n\n"):
                    para = re.sub(r"\s+", " ", para.lower())
                    for s in sent_re.split(para):
                        t = re.findall(r"[a-z][a-z'\-]*", s)
                        if len(t) >= 3:
                            fo.write(" ".join(t) + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(q):
        q.add_argument("--vocab", type=int, default=30000)
        q.add_argument("--min-count", type=int, default=20)
        q.add_argument("--dim", type=int, default=100)
        q.add_argument("--window", type=int, default=5)
        q.add_argument("--anchors", type=int, default=2000)
        q.add_argument("--cand", type=int, default=3000)
        q.add_argument("-k", type=int, default=25)
        q.add_argument("--budget", type=int, default=0, help="tokens per model (0 = largest feasible)")
        q.add_argument("--runs", type=int, default=3)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--out", default="result.json")
    q = sub.add_parser("semeval"); common(q)
    q.add_argument("--root", required=True); q.add_argument("--dedup", action="store_true")
    q.set_defaults(f=cmd_semeval)
    q = sub.add_parser("paired"); common(q)
    for n in ("gen-p1", "gen-p2", "stat-p1", "stat-p2"):
        q.add_argument("--" + n, required=True)
    q.add_argument("--dedup-stat", action="store_true")
    q.add_argument("--min-target-count", type=int, default=50)
    q.set_defaults(f=cmd_paired)
    q = sub.add_parser("prep"); q.add_argument("--inp", required=True); q.add_argument("--out", required=True)
    q.set_defaults(f=cmd_prep)
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
