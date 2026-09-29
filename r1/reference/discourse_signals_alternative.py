"""Coherence and semantic shift from an alternative sentence representation (reference only).

Requires the restricted sentence file (columns: isbn, sentence_list), which is not distributed.
The definitions are those of coherence_topic_calc.py; only the sentence representation changes.

  python discourse_signals_alternative.py --in sentences.csv --measure krsbert --out discourse_krsbert.jsonl
  python discourse_signals_alternative.py --in sentences.csv --measure lexical --out discourse_lexical.jsonl
"""
import argparse, ast, json
from collections import Counter
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--measure", choices=["krsbert", "lexical"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

if a.measure == "krsbert":
    import torch
    from transformers import AutoTokenizer, AutoModel
    MODEL_ID = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()

    @torch.no_grad()
    def represent(sents, batch_size=64, max_length=128):
        """Mean-pooled sentence embeddings."""
        out = []
        for i in range(0, len(sents), batch_size):
            enc = tok(sents[i:i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            h = model(**enc)[0]
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy())
        return np.concatenate(out).astype(np.float64)
else:
    def represent(sents):
        """Character-bigram count vectors."""
        counts = [Counter(s[j:j + 2] for j in range(len(s) - 1)) for s in sents]
        vocab = {}
        for c in counts:
            for b in c:
                vocab.setdefault(b, len(vocab))
        V = np.zeros((len(sents), max(1, len(vocab))))
        for t, c in enumerate(counts):
            for b, n in c.items():
                V[t, vocab[b]] = n
        return V


def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else 0.0


def signals(V):
    """Coherence: cosine with the previous sentence. Semantic shift: one minus the cosine with the mean
    of all previous sentences. The first sentence is 0 for both."""
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(norm == 0, 1, norm)
    coh, shift = [0.0] * len(V), [0.0] * len(V)
    for t in range(1, len(V)):
        coh[t] = round(cosine(V[t - 1], V[t]), 4)
        shift[t] = round(1.0 - cosine(V[t], V[:t].mean(axis=0)), 4)
    return coh, shift


df = pd.read_csv(a.inp, usecols=["isbn", "sentence_list"])
if a.limit:
    df = df.head(a.limit)
with open(a.out, "w") as f:
    for row, (isbn, s) in enumerate(zip(df.isbn, df.sentence_list)):
        coh, shift = signals(represent(ast.literal_eval(s)))
        f.write(json.dumps({"row": row, "isbn": str(isbn), "coherence": coh, "semantic_shift": shift}) + "\n")
