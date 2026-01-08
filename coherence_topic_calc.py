import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==== 1. Load Korean SBERT (Using safetensors) ====
EMB_MODEL_ID = "jhgan/ko-sroberta-multitask"
print(f"Loading sentence embedding model (HF): {EMB_MODEL_ID}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
emb_model = AutoModel.from_pretrained(
    EMB_MODEL_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
)
emb_model.to(device)
emb_model.eval()
print("Embedding model loaded.")

# ==== 2. Sentence Splitting: Reuse Existing Logic ====
# Ensure 'calculate_surprisal.py' is in the same directory.
from calculate_surprisal import split_sentences_combined, tokenizer

# ==== 3. Mean Pooling Function ====
def mean_pooling(model_output, attention_mask):
    """
    Standard HF implementation: Performs mean pooling considering the attention mask.
    """
    token_embeddings = model_output[0]              # (Batch, Time_steps, Dimension)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)  # (Batch, Dimension)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)             # (Batch, Dimension)
    return sum_embeddings / sum_mask

# ==== 4. SBERT-based Embedding Generation ====
@torch.no_grad()
def encode_sentences_sbert(sent_list, batch_size=64, max_length=128):
    """
    Directly loads ko-sroberta-multitask via HF to generate L2-normalized sentence embeddings.
    """
    if not sent_list:
        dim = emb_model.config.hidden_size
        return np.empty((0, dim), dtype="float32")

    all_embs = []
    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i:i + batch_size]
        enc = emb_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

    # Output generation and mean pooling
    outputs = emb_model(**enc)
    sent_emb = mean_pooling(outputs, enc["attention_mask"])  # (B, D)
    
    # L2 normalization to enable cosine similarity via dot product
    sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
    all_embs.append(sent_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)  # (N, D)

# ==== 5. Calculate Coherence and Semantic Shift ====
def calculate_coherence_and_semantic_shift(sentences):
    """
    Computes local coherence (adjacent similarity) and global semantic shift (deviation from context).
    """
    if not sentences:
        return [], []

    embs = encode_sentences_sbert(sentences)  # (N, D)
    n = embs.shape[0]

    coherence = [0.0] * n
    semantic_shift = [0.0] * n

    # Local Coherence: $cos(v_t, v_{t-1})$
    # Since embeddings are L2 normalized, dot product equals cosine similarity.
    for t in range(1, n):
        a = embs[t - 1]
        b = embs[t]
        cos = float(np.dot(a, b))
        coherence[t] = round(cos, 4)

    # Semantic Shift: $1 - cos(v_t, \mu_{1..t-1})$
    # Measures how much the current sentence deviates from the cumulative context mean.
    for t in range(1, n):
        mean_prev = embs[:t].mean(axis=0)
        norm = np.linalg.norm(mean_prev) + 1e-9
        mean_prev = mean_prev / norm
        v = embs[t]
        cos = float(np.dot(v, mean_prev))
        dist = 1.0 - cos
        semantic_shift[t] = round(dist, 4)

    return coherence, semantic_shift

# ==== 6. Main: Merge with Existing Surprisal CSV ====
def main():
    base_path = "/home/qgroup2/sungpil/short_novel"
    input_file = os.path.join(base_path, "short_novel_with_surprisal.csv")
    
    # Updated output filename to reflect additional metrics
    output_file = os.path.join(base_path, "short_novel_with_surprisal_coherence_semantic.csv")

    print(f"Reading CSV from: {input_file}")
    if not os.path.exists(input_file):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_file)
    if 'text' not in df.columns:
        print("Error: CSV must contain a 'text' column.")
        return

    print(f"Total rows to process: {len(df)}")

    all_sentence_lists = []
    all_coherence_vectors = []
    all_semantic_shift_vectors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Coherence/Semantic"):
        try:
            text = row['text']
            sentences = split_sentences_combined(text, tokenizer)
            
            # Compute vectors
            coherence_vec, semantic_shift_vec = calculate_coherence_and_semantic_shift(sentences)

            all_sentence_lists.append(str(sentences))
            all_coherence_vectors.append(str(coherence_vec))
            all_semantic_shift_vectors.append(str(semantic_shift_vec))

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            all_sentence_lists.append("[]")
            all_coherence_vectors.append("[]")
            all_semantic_shift_vectors.append("[]")

    # Store results in DataFrame
    df['sentence_list'] = all_sentence_lists
    df['coherence_vector'] = all_coherence_vectors
    df['semantic_shift_vector'] = all_semantic_shift_vectors

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("=" * 40)
    print("Processing Complete!")
    print(f"Saved to: {output_file}")
    print("=" * 40)

if __name__ == "__main__":
    main()