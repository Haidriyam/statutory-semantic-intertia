"""Experiment 1: one fixed language, R independent corpus draws."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from sim import *

R = 20
P_CHANGE = 0.30
LANG_SEED = 20260101
out_path = sys.argv[1] if len(sys.argv) > 1 else "exp1_results.json"

lang = Language(LANG_SEED)
rng0 = np.random.default_rng(LANG_SEED + 1)
g_all, topic2 = lang.change_spec(P_CHANGE, rng0)       # fixed across draws
M1, M2 = lang.topic_matrix(), lang.topic_matrix(g_all, topic2)

draws = []
for r in range(R):
    base = 1000 * (r + 1)
    m1 = build_model(lang, M1, N_SENT, base + 1)
    m2 = build_model(lang, M2, N_SENT, base + 2)
    mn = build_model(lang, M1, N_SENT, base + 3)       # independent period-1 null
    draws.append((m1, m2, mn))
    print("draw", r, flush=True)

valid_all = np.all([np.all([m.freq >= MINCOUNT for m in d], axis=0) for d in draws], axis=0)
valid_all[:NF] = False
common_ids = np.where(valid_all)[0]


def run_draw(m1, m2, mn, k=25, anchors="top200"):
    tg = valid_targets([m1, m2, mn])
    cand = top_words([m1, m2, mn], 600)
    if anchors == "func60":
        anc = np.arange(NF)
    else:
        anc = top_words([m1, m2, mn], int(anchors[3:]))
    D = displacement(m1, m2, anc)
    D0 = displacement(m1, mn, anc)
    return tg, cand, D, D0


res = {"R": R, "p": P_CHANGE}
keys = ["D", "ED", "J25", "EJ", "I05", "I025", "I075", "I0", "I1", "LSI", "J10", "J50"]
auc_d = {k: [] for k in keys}
afunc, a200, a600 = [], [], []
rho_g_D, rho_g_J, rho_DJ = [], [], []
fc = {"D": [], "ED": [], "J": [], "EJ": []}
null_mean_dec = []; null_q95_dec = []; mean_D_changed = []
fpr, tpr = [], []
mean_D_unch, mean_D0_unch = [], []
bins = [(0, 0), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
dose_D = {i: [] for i in range(5)}; dose_J = {i: [] for i in range(5)}
store_D, store_D0, store_J = {}, {}, {}

for r, (m1, m2, mn) in enumerate(draws):
    tg, cand, D, D0 = run_draw(m1, m2, mn)
    Dt, D0t = D[tg], D0[tg]
    J = jaccard(m1, m2, tg, cand, 25)
    J0 = jaccard(m1, mn, tg, cand, 25)
    g = g_all[tg - NF]
    lab = (g > 0).astype(int)
    band = deciles(m1.freq[tg])
    ED = Dt - band_mean(D0t, band)[band]
    EJ = band_mean(J0, band)[band] - J
    cosine = 1 - Dt
    S = {"D": Dt, "ED": ED, "J25": -J, "EJ": EJ,
         "I05": -(0.5 * cosine + 0.5 * J), "I025": -(0.25 * cosine + 0.75 * J),
         "I075": -(0.75 * cosine + 0.25 * J), "I0": -J, "I1": -cosine,
         "LSI": -(cosine * (1 + J) / 2),
         "J10": -jaccard(m1, m2, tg, cand, 10), "J50": -jaccard(m1, m2, tg, cand, 50)}
    for k in keys:
        auc_d[k].append(auc(S[k], lab))
    # alignment variants
    for name, store in (("func60", afunc), ("top200", a200), ("top600", a600)):
        _, _, Da, _ = run_draw(m1, m2, mn, anchors=name)
        store.append(auc(Da[tg], lab))
    rho_g_D.append(spearmanr(Dt, g)[0]); rho_g_J.append(spearmanr(-J, g)[0])
    rho_DJ.append(spearmanr(Dt, J)[0])
    un = lab == 0
    lf = np.log(m1.freq[tg])
    fc["D"].append(spearmanr(Dt[un], lf[un])[0]); fc["ED"].append(spearmanr(ED[un], lf[un])[0])
    fc["J"].append(spearmanr(J[un], lf[un])[0]); fc["EJ"].append(spearmanr(EJ[un], lf[un])[0])
    null_mean_dec.append(band_mean(D0t, band)); null_q95_dec.append(band_quantile(D0t, band, 0.95))
    mean_D_changed.append(Dt[~un].mean())
    thr = band_quantile(D0t, band, 0.95)[band]
    flag = Dt > thr
    tpr.append(flag[~un].mean()); fpr.append(flag[un].mean())
    mean_D_unch.append(Dt[un].mean()); mean_D0_unch.append(D0t[un].mean())
    for i, (lo, hi) in enumerate(bins):
        m = (g == 0) if i == 0 else ((g > 0) & (g <= hi) if i == 1 else (g > lo) & (g <= hi))
        dose_D[i].append(Dt[m].mean()); dose_J[i].append(J[m].mean())
    ids = tg
    store_D[r] = dict(zip(ids.tolist(), Dt)); store_D0[r] = dict(zip(ids.tolist(), D0t)); store_J[r] = dict(zip(ids.tolist(), J))


def ms(x):
    x = np.asarray(x, float)
    return [float(np.nanmean(x)), float(np.nanstd(x, ddof=1))]


res["auc"] = {k: ms(v) for k, v in auc_d.items()}
res["auc_alignment"] = {"func60": ms(afunc), "top200": ms(a200), "top600": ms(a600)}
res["spearman_g"] = {"D": ms(rho_g_D), "negJ": ms(rho_g_J)}
res["spearman_D_J"] = ms(rho_DJ)
res["freq_corr_unchanged"] = {k: ms(v) for k, v in fc.items()}
res["null_mean_by_decile"] = np.nanmean(null_mean_dec, 0).tolist()
res["null_q95_by_decile"] = np.nanmean(null_q95_dec, 0).tolist()
res["mean_D_changed"] = ms(mean_D_changed)
res["per_word_rule"] = {"tpr": ms(tpr), "fpr": ms(fpr)}
res["mean_D_unchanged_in_changing"] = ms(mean_D_unch)
res["mean_D_unchanged_null"] = ms(mean_D0_unch)
res["dose_D"] = [ms(dose_D[i]) for i in range(5)]
res["dose_J"] = [ms(dose_J[i]) for i in range(5)]
res["dose_bins"] = ["0", "(0,0.5]", "(0.5,0.7]", "(0.7,0.85]", "(0.85,1]"]

# reliability on words valid in all draws
ids = [i for i in common_ids.tolist() if all(i in store_D[r] for r in range(R))]
XD = np.array([[store_D[r][i] for r in range(R)] for i in ids])
XJ = np.array([[store_J[r][i] for r in range(R)] for i in ids])
X0 = np.array([[store_D0[r][i] for r in range(R)] for i in ids])
unch = np.array([g_all[i - NF] == 0 for i in ids])
res["reliability"] = {
    "n_words": len(ids),
    "spearman_D": mean_pairwise_spearman(XD), "spearman_J": mean_pairwise_spearman(XJ),
    "icc_D": icc1(XD), "icc_J": icc1(XJ),
    "spearman_D_unchanged": mean_pairwise_spearman(XD[unch])[0],
    "spearman_null_unchanged": mean_pairwise_spearman(X0[unch])[0]}

json.dump(res, open(out_path, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
