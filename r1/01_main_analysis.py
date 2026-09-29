"""Main-settings results of the revised manuscript: Table 1, Table 2 (with Supplementary Tables S5, S6),
Table 3, translation comparison (S1d), variance shares and three-signal clustering (Section 3.2),
choice of k (Supplementary Fig. S3), Table 4 and the position-matched null (Section 3.3.1),
Table 5 (with S11, S12). Outputs: out/main_*.csv"""
import numpy as np, pandas as pd
import scikit_posthocs as sp
from kneed import KneeLocator
from scipy.stats import (chi2_contingency, kruskal, levene, mannwhitneyu, pearsonr, shapiro, ttest_1samp)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from common import *

df, S, C, M, lab = load_master()
N = len(S)
X = standardise(curves(S)); Cz = standardise(curves(C)); Mz = standardise(curves(M))

# archetype labels are those of k-means (k = 5, random_state 42, n_init 20) on X
assert adjusted_rand_score(lab, kmeans_labels(X)) == 1.0

# ---- Table 1: within-story correlations, first sentence excluded
rows = []
for name, (a, b) in {"surprisal-coherence": (S, C), "surprisal-shift": (S, M), "coherence-shift": (C, M)}.items():
    r = np.array([pearsonr(x[1:], y[1:])[0] if np.std(x[1:]) > 0 and np.std(y[1:]) > 0 else 0.0 for x, y in zip(a, b)])
    rows.append(dict(pair=name, mean=r.mean(), median=np.median(r), sd=r.std(ddof=1),
                     q25=np.percentile(r, 25), q75=np.percentile(r, 75)))
pd.DataFrame(rows).to_csv(f"{OUT}/main_table1_correlations.csv", index=False)

# ---- peaks, Table 3, shape descriptors (Table 2, S5, S6)
peaks = [find_peaks(z, height=1.0)[0] for z in X]
count = np.array([len(p) for p in peaks], float)
position = np.array([p[np.argmax(z[p])] / (X.shape[1] - 1) if len(p) else np.nan for p, z in zip(peaks, X)])
intensity = np.array([z[p].max() if len(p) else np.nan for p, z in zip(peaks, X)])
rows = []
for k in range(K):
    c = count[lab == k]
    d = dict(archetype=k, n=len(c), mean=c.mean(), sd=c.std(ddof=1), multi_peak_pct=100 * (c >= 2).mean(),
             n_with_peak=int((c > 0).sum()), position_mean=np.nanmean(position[lab == k]),
             intensity_mean=np.nanmean(intensity[lab == k]))
    d.update({f"pct_{j}_peaks": 100 * (c == j).mean() for j in range(5)})
    rows.append(d)
pd.DataFrame(rows).to_csv(f"{OUT}/main_table3_peak_counts.csv", index=False)
rows = []
for name, v in [("position", position), ("intensity", intensity), ("count", count)]:
    gs = [v[(lab == k) & np.isfinite(v)] for k in range(K)]
    H, p = kruskal(*gs)
    rows.append(dict(descriptor=name, N=sum(len(g) for g in gs), H=H, p=p,
                     shapiro_min_p=min(shapiro(g).pvalue for g in gs), levene_p=levene(*gs).pvalue))
    sub = pd.DataFrame(dict(a=lab, v=v)).dropna()
    sp.posthoc_dunn(sub, val_col="v", group_col="a", p_adjust="bonferroni").to_csv(f"{OUT}/main_S6_dunn_{name}.csv")
pd.DataFrame(rows).to_csv(f"{OUT}/main_table2_S5_descriptors.csv", index=False)

# ---- translation status (S1d)
origin = df.country.apply(lambda c: np.nan if pd.isna(c) else ("Korean" if c == "한국" else "Translated"))
ok = origin.notna().values
ct = pd.crosstab(lab[ok], origin[ok])
chi2, p, dof, exp = chi2_contingency(ct)
ct.rename_axis("archetype").to_csv(f"{OUT}/main_S1d_crosstab.csv")
res = (ct.values - exp) / np.sqrt(exp)
a4 = (lab == 4) & ok & np.isfinite(position)
ko, tr = position[a4 & (origin == "Korean").values], position[a4 & (origin == "Translated").values]
u = mannwhitneyu(ko, tr, alternative="two-sided")
pd.DataFrame([dict(chi2=chi2, dof=dof, p=p, residual_A4_translated=res[4, list(ct.columns).index("Translated")],
                   translated_pct_analytical_sample=100 * (origin[ok] == "Translated").mean(),
                   U=u.statistic, U_p=u.pvalue, n_korean=len(ko), n_translated=len(tr),
                   mean_position_korean=ko.mean(), mean_position_translated=tr.mean())]
             ).to_csv(f"{OUT}/main_S1d_translation.csv", index=False)

# ---- how far the partition is reflected in the discourse signals (Section 3.2)
def variance_share(Z, labels):
    g = Z.mean(0)
    return sum((labels == k).sum() * ((Z[labels == k].mean(0) - g) ** 2).sum() for k in range(K)) / ((Z - g) ** 2).sum()
three = kmeans_labels(np.hstack([X, Cz, Mz]))
pd.DataFrame([dict(share_coherence=variance_share(Cz, lab), share_shift=variance_share(Mz, lab),
                   share_surprisal=variance_share(X, lab), ARI_three_signal_clustering=adjusted_rand_score(lab, three))]
             ).to_csv(f"{OUT}/main_partition_vs_discourse.csv", index=False)

# ---- choice of k (Supplementary Fig. S3) and reproducibility across seeds
rows, inertia = [], []
for k in range(2, 13):
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
    inertia.append(km.inertia_)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_)))
ks = pd.DataFrame(rows)
ks["elbow"] = KneeLocator(list(range(2, 13)), inertia, curve="convex", direction="decreasing").knee
ks.to_csv(f"{OUT}/main_k_scan.csv", index=False)
seed_ari = [adjusted_rand_score(lab, KMeans(n_clusters=K, random_state=s, n_init=20).fit_predict(X)) for s in range(10, 20)]
pd.DataFrame(dict(seed=range(10, 20), ARI=seed_ari)).to_csv(f"{OUT}/main_seed_ari.csv", index=False)

# ---- Table 4: deviations at surprisal peaks; position-matched null
story, pos = peak_events(X, 1.0)
vc, vm, g = Cz[story, pos], Mz[story, pos], lab[story]
rng = np.random.default_rng(0)
R = 1000
null_c, null_m = np.empty((R, K + 1)), np.empty((R, K + 1))
for r in range(R):
    j = rng.integers(0, N - 1, size=len(story)); j = j + (j >= story)       # a different story, same position
    nc, nm = Cz[j, pos], Mz[j, pos]
    null_c[r] = [nc.mean()] + [nc[g == k].mean() for k in range(K)]
    null_m[r] = [nm.mean()] + [nm[g == k].mean() for k in range(K)]
rows = []
for gi, name in enumerate(["pooled"] + [f"A{k}" for k in range(K)]):
    sel = np.ones(len(g), bool) if gi == 0 else g == gi - 1
    f = 1 if gi == 0 else BONF
    for sig, v, nul in [("coherence", vc, null_c[:, gi]), ("semantic_shift", vm, null_m[:, gi])]:
        W, p = wilcoxon_p(v[sel])
        t = ttest_1samp(v[sel], 0)
        obs = v[sel].mean()
        rows.append(dict(group=name, signal=sig, n_peaks=int(sel.sum()), mean=obs, shapiro_p=shapiro(v[sel]).pvalue,
                         wilcoxon_W=W, wilcoxon_p_adj=min(1, p * f), t=t.statistic, t_p_adj=min(1, t.pvalue * f),
                         null_mean=nul.mean(), null_z=(obs - nul.mean()) / nul.std(ddof=1),
                         null_p=(1 + np.sum(np.abs(nul - nul.mean()) >= abs(obs - nul.mean()))) / (R + 1)))
pd.DataFrame(rows).to_csv(f"{OUT}/main_table4_deviations_and_null.csv", index=False)
rows = []
for sig, v in [("coherence", vc), ("semantic_shift", vm)]:
    H, p = kw(v, g)
    rows.append(dict(signal=sig, H=H, p=p))
    sp.posthoc_dunn(pd.DataFrame(dict(a=g, v=v)), val_col="v", group_col="a", p_adjust="bonferroni"
                    ).to_csv(f"{OUT}/main_S9_dunn_{sig}.csv")
pd.DataFrame(rows).to_csv(f"{OUT}/main_S8_kw_deviations.csv", index=False)

# ---- Table 5: recovery
rec = recovery_at_events(Cz, Mz, story, pos, 10)
names = ["TTR_C", "Slope_C", "TTR_M", "Slope_M"]
rows = []
for k in range(K):
    d = dict(archetype=k)
    for j, n in enumerate(names):
        v = rec[g == k, j]; v = v[np.isfinite(v)]
        d[f"{n}_mean"], d[f"{n}_sd"], d[f"{n}_n"] = v.mean(), v.std(ddof=1), len(v)
    rows.append(d)
pd.DataFrame(rows).to_csv(f"{OUT}/main_table5_recovery.csv", index=False)
rows = []
for j, n in enumerate(names):
    H, p = kw(rec[:, j], g)
    rows.append(dict(measure=n, H=H, p=p, n_events=int(np.isfinite(rec[:, j]).sum()), n_peaks=len(story)))
    sub = pd.DataFrame(dict(a=g, v=rec[:, j])).dropna()
    sp.posthoc_dunn(sub, val_col="v", group_col="a", p_adjust="bonferroni").to_csv(f"{OUT}/main_S12_dunn_{n}.csv")
pd.DataFrame(rows).to_csv(f"{OUT}/main_S11_kw_recovery.csv", index=False)
print("main analysis written to", OUT)
