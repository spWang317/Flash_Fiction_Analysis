"""Shared functions for the analyses. Inputs are numerical signals only."""
import ast, json, math, os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import kruskal, wilcoxon, zscore
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MASTER = os.path.join(REPO, "flash_fiction_with_surprisal_coherence_semantic.csv")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
K = 5          # number of archetypes
BONF = 10      # 5 archetypes x 2 discourse signals


def load_master():
    df = pd.read_csv(MASTER)
    S = [np.array(ast.literal_eval(x), float) for x in df.surprisal_vector]
    C = [np.array(ast.literal_eval(x), float) for x in df.coherence_vector]
    M = [np.array(ast.literal_eval(x), float) for x in df.semantic_shift_vector]
    lab = pd.read_csv(os.path.join(DATA, "archetype_labels.csv"))
    assert (lab.isbn.astype(str).values == df.isbn.astype(str).values).all()
    return df, S, C, M, lab.archetype.values.astype(int)


def load_jsonl(name, keys):
    rows = [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip()]
    rows.sort(key=lambda d: d["row"])
    return [[np.array(d[k], float) for d in rows] for k in keys]


def resample(v, L=50):
    v = np.asarray(v, float)
    if len(v) <= 1:
        return np.full(L, v[0] if len(v) == 1 else np.nan)
    return np.round(np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v), 4)


def curves(vecs, L=50, sigma=2.0):
    """Resample each story to L positions and smooth with a Gaussian kernel of width sigma."""
    return np.vstack([np.round(gaussian_filter1d(resample(v, L), sigma=sigma), 4) for v in vecs])


def standardise(X, start=2):
    """Drop the positions before the stable region and z-score each story."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(zscore(X[:, start:], axis=1))


def stable_region_start(Sc, L):
    """Variance diagnostic of Section 2.2.4: largest single-step drop of the across-story variance
    within the first 20% of positions; the stable region starts at the bin after that drop."""
    keep = [c for c in Sc if np.std(c) > 0]
    Z = np.vstack([(c - c.mean()) / c.std() for c in keep])
    v = Z.var(axis=0)
    W = math.ceil(0.2 * L)
    drop = (v[:-1] - v[1:]) / v[:-1]
    return 1 + int(np.argmax(drop[:W])), v


def peak_events(Z, threshold=1.0):
    story, pos = [], []
    for i, z in enumerate(Z):
        p = find_peaks(z, height=threshold)[0]
        story += [i] * len(p); pos += p.tolist()
    return np.array(story, int), np.array(pos, int)


def recovery(series, kind):
    """Time to recovery and recovery slope within the window `series` (first element = peak position)."""
    v0 = series[0]
    for i in range(1, len(series)):
        v = series[i]
        if (kind == "coherence" and v0 < 0 and v >= 0) or (kind == "shift" and v0 > 0 and v <= 0):
            return i, (v - v0) / i
    return np.nan, np.nan


def recovery_at_events(Cz, Mz, story, pos, window=10):
    out = np.full((len(story), 4), np.nan)
    for e, (i, p) in enumerate(zip(story, pos)):
        out[e, 0:2] = recovery(Cz[i, p:p + window], "coherence")
        out[e, 2:4] = recovery(Mz[i, p:p + window], "shift")
    return out   # TTR_C, Slope_C, TTR_M, Slope_M


def wilcoxon_p(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = wilcoxon(x)
    return float(r.statistic), float(r.pvalue)


def kw(values, groups):
    gs = [values[(groups == k) & np.isfinite(values)] for k in range(K)]
    r = kruskal(*gs)
    return float(r.statistic), float(r.pvalue)


def kmeans_labels(X, k=K):
    return KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)


def match_to_reference(ref, lab):
    """Relabel `lab` so that each new cluster carries the reference label it overlaps most (one-to-one)."""
    ct = np.zeros((K, K))
    for a, b in zip(ref, lab):
        ct[a, b] += 1
    r, c = linear_sum_assignment(-ct)
    m = np.empty(K, int); m[c] = r
    return m[lab]


def mean_curves(X, labels, start, L):
    """Mean curve of each archetype and the story positions (0-1) of its points."""
    return np.arange(start, L) / (L - 1), np.vstack([X[labels == k].mean(0) for k in range(K)])


def curve_correlation(ref_pos, ref_curves, pos, new_curves, n=200):
    """Correlation between the original mean curves and the mean curves of the matched archetypes of another
    setting, both interpolated onto the story positions that the two settings share."""
    if len(pos) == len(ref_pos) and np.allclose(pos, ref_pos):      # same positions: no interpolation needed
        return np.array([np.corrcoef(ref_curves[k], new_curves[k])[0, 1] for k in range(K)])
    g = np.linspace(max(ref_pos[0], pos[0]), min(ref_pos[-1], pos[-1]), n)
    return np.array([np.corrcoef(np.interp(g, ref_pos, ref_curves[k]), np.interp(g, pos, new_curves[k]))[0, 1]
                     for k in range(K)])


def deviations_summary(Cz, Mz, story, pos, labels):
    """Pooled and per-archetype deviations of coherence and semantic shift at the peak positions."""
    vc, vm = Cz[story, pos], Mz[story, pos]
    g = labels[story]
    d = dict(n_peaks=len(story), coherence_pooled=vc.mean(), coherence_pooled_p=wilcoxon_p(vc)[1],
             shift_pooled=vm.mean(), shift_pooled_p=wilcoxon_p(vm)[1])
    pc = [min(1.0, wilcoxon_p(vc[g == k])[1] * BONF) for k in range(K)]
    pm = [min(1.0, wilcoxon_p(vm[g == k])[1] * BONF) for k in range(K)]
    d["n_archetypes_coherence_sig"] = sum(p < 0.05 for p in pc)
    d["archetypes_coherence_not_sig"] = ", ".join(f"A{k}" for k in range(K) if pc[k] >= 0.05) or "-"
    d["n_archetypes_shift_sig"] = sum(p < 0.05 for p in pm)
    return d, vc, vm, g


def recovery_summary(Cz, Mz, story, pos, labels, window=10):
    rec = recovery_at_events(Cz, Mz, story, pos, window)
    g = labels[story]
    return {f"KW_p_{n}": kw(rec[:, j], g)[1] for j, n in enumerate(["TTR_C", "Slope_C", "TTR_M", "Slope_M"])}, rec
