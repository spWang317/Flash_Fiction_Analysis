"""Sensitivity to analysis settings (Section 3.4; Supplementary Tables S13 and S14).
Outputs: out/S13_settings.csv, out/S14_clustering.csv"""
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from common import *

df, S, C, M, lab = load_master()
(Q,) = load_jsonl("surprisal_qwen25_7b.jsonl", ["surprisal"])
Q = [np.where(np.isfinite(q), q, np.nanmean(q)) for q in Q]      # two sentences have no scored token
KC, KM = load_jsonl("discourse_krsbert.jsonl", ["coherence", "semantic_shift"])
LC, LM = load_jsonl("discourse_lexical.jsonl", ["coherence", "semantic_shift"])
X0 = standardise(curves(S))                                       # main surprisal curves
REF_POS, REF_CURVES = mean_curves(X0, lab, 2, 50)


def run(name, Sv, Cv, Mv, L=50, sigma=2.0, Z=1.0, rederive=False, fixed_start=None):
    Sc = curves(Sv, L, sigma)
    start = fixed_start if fixed_start is not None else stable_region_start(Sc, L)[0]
    window = int(round(10 * L / 50))
    X, Cz, Mz = standardise(Sc, start), standardise(curves(Cv, L, sigma), start), standardise(curves(Mv, L, sigma), start)
    out = []
    modes = [("original", lab)]
    if rederive:
        modes.append(("re-derived", match_to_reference(lab, kmeans_labels(X))))
    for mode, labels in modes:
        story, pos = peak_events(X, Z)
        d, *_ = deviations_summary(Cz, Mz, story, pos, labels)
        r, _ = recovery_summary(Cz, Mz, story, pos, labels, window)
        row = dict(setting=name, labels=mode, L=L, sigma=sigma, Z=Z, window=window, stable_region_start=start, **d, **r)
        if mode == "re-derived":
            row["min_curve_correlation"] = curve_correlation(REF_POS, REF_CURVES, *mean_curves(X, labels, start, L)).min()
            row["stories_remaining_pct"] = 100 * (labels == lab).mean()
        out.append(row)
    return out


rows = run("Main settings", S, C, M, fixed_start=2)
rows += run("sigma = 1.0", S, C, M, sigma=1.0, rederive=True)
rows += run("sigma = 1.5", S, C, M, sigma=1.5, rederive=True)
rows += run("L = 60", S, C, M, L=60, sigma=2.4, rederive=True)
rows += run("L = 70", S, C, M, L=70, sigma=2.8, rederive=True)
rows += run("Z = 0.5", S, C, M, Z=0.5, fixed_start=2)
rows += run("Z = 1.5", S, C, M, Z=1.5, fixed_start=2)
rows += run("Encoder: KR-SBERT", S, KC, KM, fixed_start=2)
rows += run("Discourse measure: lexical overlap", S, LC, LM, fixed_start=2)
rows += run("Language model: Qwen2.5-7B", Q, C, M, rederive=True, fixed_start=2)
pd.DataFrame(rows).to_csv(f"{OUT}/S13_settings.csv", index=False)

# ---- alternative clustering methods (S14)
methods = {"Gaussian mixture": GaussianMixture(n_components=K, covariance_type="diag", random_state=0).fit(X0).predict(X0),
           "Spectral": SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=0).fit_predict(X0),
           "Ward": AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X0)}
rows = []
for name, raw in methods.items():
    new = match_to_reference(lab, raw)
    r = curve_correlation(REF_POS, REF_CURVES, *mean_curves(X0, new, 2, 50))
    rows.append(dict(method=name, measure="Correlation with original mean curve", **{f"A{k}": r[k] for k in range(K)}))
    rows.append(dict(method=name, measure="Stories remaining in corresponding cluster (%)",
                     **{f"A{k}": 100 * (new[lab == k] == k).mean() for k in range(K)}))
pd.DataFrame(rows).to_csv(f"{OUT}/S14_clustering.csv", index=False)
print("sensitivity results written to", OUT)
