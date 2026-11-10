"""Draw Figs. 1-3 from exp1_results.json and exp2_results_summary.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
e1 = json.load(open("exp1_results.json"))
e2 = json.load(open("exp2_results_summary.json"))

# Fig 1: dose-response
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(5)
for key, lab, mk, ls in (("dose_D", "D (cosine displacement)", "o", "-"),
                         ("dose_J", "J (neighbour Jaccard, k=25)", "s", "--")):
    m = np.array([v[0] for v in e1[key]]); s = np.array([v[1] for v in e1[key]])
    ax.errorbar(x, m, yerr=s, marker=mk, ls=ls, color="k" if key == "dose_D" else "0.45",
                capsize=2, label=lab, ms=3.5, lw=1)
ax.set_xticks(x); ax.set_xticklabels(["0", "(0,.5]", "(.5,.7]", "(.7,.85]", "(.85,1]"])
ax.set_xlabel("True graded change $g$ (injected)"); ax.set_ylabel("Mean score")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("fig1_dose_response.pdf"); fig.savefig("fig1_dose_response.png", dpi=200)

# Fig 2: null by decile
fig, ax = plt.subplots(figsize=(3.4, 2.4))
d = np.arange(1, 11)
ax.semilogy(d, e1["null_q95_by_decile"], "k^-", ms=3.5, lw=1, label="null $D$, 95th pct.")
ax.semilogy(d, e1["null_mean_by_decile"], "ko--", ms=3.5, lw=1, label="null $D$, mean")
ax.axhline(e1["mean_D_changed"][0], color="0.5", ls=":", lw=1)
ax.text(5.2, e1["mean_D_changed"][0] * 1.12, "mean $D$ of changed words", fontsize=6.5)
ax.set_xticks(d); ax.set_xlabel("Frequency decile (1 = rarest, 10 = most frequent)")
ax.set_ylabel("$D$ with no true change"); ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig("fig2_null_by_decile.pdf"); fig.savefig("fig2_null_by_decile.png", dpi=200)

# Fig 3: inference outcomes
conds = [("S1", "S1\ntrue effect"), ("S2", "S2\nno effect"),
         ("S3full", "S3\ncopy-fwd,\nfull"), ("S3dedup", "S3\ncopy-fwd,\ndedup")]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
w = 0.26
for i, (k, lab, hatch, col) in enumerate((("frac_excl0_rawD", "raw $D$", "", "0.75"),
                                          ("frac_excl0_ED", "calibrated $D$", "", "0.35"),
                                          ("frac_excl0_EJ", "calibrated $J$", "///", "white"))):
    ax.bar(np.arange(4) + (i - 1) * w, [e2[c][k] for c, _ in conds], w,
           color=col, edgecolor="k", hatch=hatch, lw=0.6, label=lab)
ax.axhline(0.05, color="k", ls=":", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in conds], fontsize=6.5)
ax.set_ylabel("Fraction of worlds,\n95% CI excluding 0"); ax.set_ylim(0, 1.25)
ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center")
fig.tight_layout(); fig.savefig("fig3_inference.pdf"); fig.savefig("fig3_inference.png", dpi=200)
print("figures written")
