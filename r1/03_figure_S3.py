"""Supplementary Fig. S3 from out/main_k_scan.csv (run 01_main_analysis.py first)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from common import OUT
d = pd.read_csv(f"{OUT}/main_k_scan.csv")
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(9.16, 5.35), dpi=200)
l1, = ax.plot(d.k, d.inertia, "o-", color="tab:blue", lw=2.5, ms=8, label="Inertia")
ax.set_xlabel("Number of clusters (k)", fontsize=17); ax.set_ylabel("Inertia (SSE)", color="tab:blue", fontsize=17)
ax.tick_params(axis="y", labelcolor="tab:blue")
l2 = ax.axvline(5, color="red", ls="--", lw=2, label="k = 5")
ax2 = ax.twinx()
l3, = ax2.plot(d.k, d.silhouette, "s-", color="tab:orange", lw=2.5, ms=8, label="Silhouette score")
ax2.set_ylabel("Silhouette score", color="tab:orange", fontsize=17); ax2.tick_params(axis="y", labelcolor="tab:orange")
ax.grid(True, ls=":", alpha=.6); ax.legend(handles=[l1, l2, l3], loc="upper right", fontsize=15)
fig.tight_layout(); fig.savefig(f"{OUT}/FigS3_k_selection.png")
