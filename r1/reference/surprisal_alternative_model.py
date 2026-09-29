"""Sentence-level surprisal from an alternative causal language model (reference only).

Requires the restricted sentence file (columns: row, isbn, sentence_list), which is not distributed.
The sentence segmentation is held fixed; only the language model changes.

The computation follows calculate_surprisal.py: for sentence i the input is the preceding sentences
followed by sentence i, the surprisal is the mean negative log-likelihood over the tokens of sentence i,
special tokens are added only before the first sentence, and the history is truncated to the last
WINDOW tokens. When a whole story fits within WINDOW tokens, this equals a single forward pass over
the concatenated story, which is what the script does; longer stories are processed sentence by sentence.

Rows already present in --out are skipped, so an interrupted run can be resumed.

  python surprisal_alternative_model.py --in sentences.csv --model Qwen/Qwen2.5-7B --out surprisal_qwen25_7b.jsonl
"""
import argparse, ast, json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=3500)
ap.add_argument("--nf4", action="store_true")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--start", type=int, default=0, help="first row (inclusive)")
ap.add_argument("--end", type=int, default=None, help="last row (exclusive)")
ap.add_argument("--tokenizer", default=None, help="defaults to --model")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
import transformers
_dt = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
kw = {_dt: torch.bfloat16, "trust_remote_code": True}
if torch.cuda.is_available():
    kw["device_map"] = "auto"
if a.nf4:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
# special tokens the tokenizer adds in front of the first sentence (added only for the first sentence)
first_prefix = tok.encode("", add_special_tokens=True)

df = pd.read_csv(a.inp, usecols=["row", "isbn", "sentence_list"])
df = df[(df.row >= a.start) & (df.row < (a.end if a.end is not None else len(df) + 10**9))]
if a.limit:
    df = df.head(a.limit)
done = set()
if os.path.exists(a.out):
    with open(a.out) as f:
        done = {json.loads(l)["row"] for l in f if l.strip()}
print(f"model={a.model} dtype={next(model.parameters()).dtype} rows={len(df)} already_done={len(done)}", flush=True)


@torch.no_grad()
def nll_positions(ids, chunk=512):
    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    logits = model(x).logits[0, :-1]          # keep model dtype; upcast chunk by chunk
    tgt = x[0, 1:]
    out = []
    for s in range(0, logits.shape[0], chunk):
        lp = torch.log_softmax(logits[s:s + chunk].float(), -1)
        out.append(-lp.gather(1, tgt[s:s + chunk, None])[:, 0])
    return torch.cat(out).cpu().numpy()       # element i = NLL of token i+1


t0 = time.time(); n = 0
with open(a.out, "a") as f:
    for _, row in df.iterrows():
        if int(row.row) in done:
            continue
        sents = ast.literal_eval(row.sentence_list)
        pieces = [tok.encode(s, add_special_tokens=False) for s in sents]
        pieces[0] = first_prefix + pieces[0]
        out, n_long = [], 0
        if sum(len(p) for p in pieces) <= a.window:
            nll = nll_positions([t for p in pieces for t in p])
            pos = 0
            for j, p in enumerate(pieces):
                start = pos + (len(first_prefix) if j == 0 else 0)
                idx = np.arange(max(start, 1), pos + len(p)) - 1
                out.append(round(float(nll[idx].mean()), 4) if len(idx) else float("nan"))
                pos += len(p)
        else:  # sentence by sentence, with truncated history
            n_long = 1
            hist = []
            for j, p in enumerate(pieces):
                ids = hist + p
                nll = nll_positions(ids)
                start = len(hist) + (len(first_prefix) if j == 0 else 0)
                idx = np.arange(max(start, 1), len(ids)) - 1
                out.append(round(float(nll[idx].mean()), 4) if len(idx) else float("nan"))
                hist = (hist + p)[-a.window:]
        f.write(json.dumps({"row": int(row.row), "isbn": str(row.isbn), "long": n_long, "surprisal": out}) + "\n")
        f.flush(); n += 1
        if n % 50 == 0:
            print(f"{n} stories  {time.time()-t0:.0f}s", flush=True)
print(f"done {n} new stories in {time.time()-t0:.0f}s", flush=True)
