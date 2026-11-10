"""
Experiment 2: paired 'general' vs 'statutory' arms in N_WORLDS independent worlds.
Conditions
  S1 true effect  : p_gen=0.30, p_stat=0.10
  S2 no effect    : p_gen=p_stat=0.20
  S3 copy-forward : as S2, but half of the statutory period-2 sentences are copies
                    of period-1 sentences ('full') vs. copies replaced by fresh
                    period-2 sentences at equal token budget ('dedup')
  S4 size         : as S2, statutory arm trained on one third of the sentences
Statistic: paired mean over shared target words of E_gen - E_stat, E = calibrated
excess displacement (or excess Jaccard change); 95% word-level percentile bootstrap.
"""
import json, sys
import numpy as np
from scipy.stats import pearsonr
from sim import *

N_WORLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
out_path = sys.argv[1] if len(sys.argv) > 1 else "exp2_results.json"
N_BOOT = 1000


def arm_models(lang, p, rng, n_sent, seed, copy_frac=None, dedup=False):
    """Return (m1, m2, null, g, extra) for one arm; g is true change per content word."""
    g, t2 = lang.change_spec(p, rng)
    M1, M2 = lang.topic_matrix(), lang.topic_matrix(g, t2)
    r1 = np.random.default_rng(seed + 1)
    tok1 = sample_corpus(lang, M1, n_sent, r1)
    m1 = build_model_from_tokens(tok1, seed + 11)
    tok2 = sample_corpus(lang, M2, n_sent, np.random.default_rng(seed + 2))
    m2 = build_model_from_tokens(tok2, seed + 12)
    mn = build_model(lang, M1, n_sent, seed + 3)
    extra = {}
    if copy_frac is not None:
        n_copy = int(copy_frac * n_sent)
        tokc = np.concatenate([tok2[: n_sent - n_copy], tok1[:n_copy]])   # copy-forward
        extra["full"] = build_model_from_tokens(tokc, seed + 13)
        # dedup: copies removed, fresh period-2 sentences added back (equal budget)
        tokf = sample_corpus(lang, M2, n_copy, np.random.default_rng(seed + 4))
        tokd = np.concatenate([tok2[: n_sent - n_copy], tokf])
        extra["dedup"] = build_model_from_tokens(tokd, seed + 14)
    return m1, m2, mn, g, extra


def arm_scores(m1, m2, mn, anchors, cand, k=25):
    D = displacement(m1, m2, anchors)
    D0 = displacement(m1, mn, anchors)
    return D, D0


def calibrated(m1, m2, mn, tg, anchors, cand):
    D, D0 = arm_scores(m1, m2, mn, anchors, cand)
    J = jaccard(m1, m2, tg, cand, 25)
    J0 = jaccard(m1, mn, tg, cand, 25)
    band = deciles(m1.freq[tg])
    ED = D[tg] - band_mean(D0[tg], band)[band]
    EJ = band_mean(J0, band)[band] - J
    return {"rawD": D[tg], "ED": ED, "EJ": EJ}


def paired(models_gen, models_stat, g_gen, g_stat, anchors_c, anchors_gen, anchors_stat, rng):
    """Paired comparison on words valid in every model of both arms."""
    allm = list(models_gen) + list(models_stat)
    tg = valid_targets(allm)
    cand = top_words(allm, 600)
    ac = top_words(allm, 200) if anchors_c is None else anchors_c
    out = {}
    for tag, (a_g, a_s) in {"common": (ac, ac),
                            "domain": (top_words(list(models_gen), 200),
                                       top_words(list(models_stat), 200))}.items():
        sg = calibrated(*models_gen, tg, a_g, cand)
        ss = calibrated(*models_stat, tg, a_s, cand)
        out[tag] = {k: sg[k] - ss[k] for k in ("rawD", "ED", "EJ")}
    realised = g_gen[tg - NF].mean() - g_stat[tg - NF].mean()
    res = {"realised": float(realised), "n": int(len(tg))}
    idx = rng.integers(0, len(tg), (N_BOOT, len(tg)))
    for tag in out:
        for k, d in out[tag].items():
            bs = d[idx].mean(1)
            res[f"{tag}_{k}"] = {"mean": float(d.mean()),
                                 "lo": float(np.quantile(bs, 0.025)),
                                 "hi": float(np.quantile(bs, 0.975)),
                                 "se": float(bs.std(ddof=1))}
    return res


worlds = []
for w in range(N_WORLDS):
    seed = 10_000 * (w + 1)
    lang = Language(seed)
    rng = np.random.default_rng(seed + 7)
    rec = {}
    # S1
    gen = arm_models(lang, 0.30, rng, N_SENT, seed + 100)
    sta = arm_models(lang, 0.10, rng, N_SENT, seed + 200)
    rec["S1"] = paired(gen[:3], sta[:3], gen[3], sta[3], None, None, None, rng)
    # S2-S4 share the arms
    gen = arm_models(lang, 0.20, rng, N_SENT, seed + 300)
    sta = arm_models(lang, 0.20, rng, N_SENT, seed + 400, copy_frac=0.5)
    rec["S2"] = paired(gen[:3], sta[:3], gen[3], sta[3], None, None, None, rng)
    full = (sta[0], sta[4]["full"], sta[2])
    dedup = (sta[0], sta[4]["dedup"], sta[2])
    rec["S3full"] = paired(gen[:3], full, gen[3], sta[3], None, None, None, rng)
    rec["S3dedup"] = paired(gen[:3], dedup, gen[3], sta[3], None, None, None, rng)
    small = arm_models(lang, 0.20, np.random.default_rng(seed + 8), N_SENT // 3, seed + 500)
    # S4 statutory arm: same changed set as S2 would be ideal; independent draw here
    rec["S4"] = paired(gen[:3], small[:3], gen[3], small[3], None, None, None, rng)
    worlds.append(rec)
    print("world", w, flush=True)

json.dump(worlds, open(out_path, "w"), default=float)

# ---- summary ----------------------------------------------------------------
def excl0(r):
    return float(r["lo"] > 0 or r["hi"] < 0)

summary = {}
for cond in ["S1", "S2", "S3full", "S3dedup", "S4"]:
    rows = [w[cond] for w in worlds]
    summary[cond] = {
        "mean_dED": float(np.mean([r["common_ED"]["mean"] for r in rows])),
        "mean_dRawD": float(np.mean([r["common_rawD"]["mean"] for r in rows])),
        "mean_realised": float(np.mean([r["realised"] for r in rows])),
        "frac_excl0_rawD": float(np.mean([excl0(r["common_rawD"]) for r in rows])),
        "frac_excl0_ED": float(np.mean([excl0(r["common_ED"]) for r in rows])),
        "frac_excl0_EJ": float(np.mean([excl0(r["common_EJ"]) for r in rows])),
        "frac_excl0_ED_domain": float(np.mean([excl0(r["domain_ED"]) for r in rows])),
        "mean_dED_domain": float(np.mean([r["domain_ED"]["mean"] for r in rows])),
        "frac_correct_sign_ED": float(np.mean([r["common_ED"]["lo"] > 0 for r in rows])),
    }

# realised-effect tracking in S2 (changed sets differ at random between arms)
rows = [w["S2"] for w in worlds]
real = np.array([r["realised"] for r in rows])
track = {"sd_realised": float(real.std(ddof=1))}
for key in ("rawD", "ED", "EJ"):
    est = np.array([r[f"common_{key}"]["mean"] for r in rows])
    x = real if key != "EJ" else real     # E_gen - E_stat is increasing in true change
    b, a = np.polyfit(x, est, 1)
    pred = a + b * x
    lo = np.array([r[f"common_{key}"]["lo"] for r in rows])
    hi = np.array([r[f"common_{key}"]["hi"] for r in rows])
    track[key] = {"r": float(pearsonr(x, est)[0]),
                  "slope": float(b),
                  "resid_sd": float((est - pred).std(ddof=2)),
                  "mean_boot_se": float(np.mean([r[f"common_{key}"]["se"] for r in rows])),
                  "ci_contains_pred": float(np.mean((lo <= pred) & (pred <= hi))),
                  "frac_excl_pred": float(np.mean((pred < lo) | (pred > hi)))}
summary["tracking_S2"] = track
json.dump(summary, open(out_path.replace(".json", "_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
