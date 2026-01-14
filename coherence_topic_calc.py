import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. Load Korean SBERT Model
# ==========================================
EMB_MODEL_ID = "jhgan/ko-sroberta-multitask"
print(f"Loading sentence embedding model: {EMB_MODEL_ID}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
emb_model = AutoModel.from_pretrained(
    EMB_MODEL_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
)
emb_model.to(device)
emb_model.eval()

# Import the shared sentence splitting logic from previous steps
from calculate_surprisal import split_sentences_combined, tokenizer

# ==========================================
# 2. Embedding Utilities
# ==========================================
def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings considering the attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

@torch.no_grad()
def encode_sentences_sbert(sent_list, batch_size=64, max_length=128):
    """Generates L2-normalized sentence embeddings using ko-sroberta."""
    if not sent_list:
        return np.empty((0, emb_model.config.hidden_size), dtype="float32")

    all_embs = []
    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i:i + batch_size]
        enc = emb_tokenizer(
            batch, padding=True, truncation=True, 
            max_length=max_length, return_tensors="pt"
        ).to(device)

        outputs = emb_model(**enc)
        sent_emb = mean_pooling(outputs, enc["attention_mask"])
        
        # Normalize to unit length for dot-product cosine similarity
        sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
        all_embs.append(sent_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)

# ==========================================
# 3. Discourse Signal Calculation
# ==========================================
def calculate_coherence_and_semantic_shift(sentences):
    """Computes local coherence and cumulative semantic shift vectors."""
    if not sentences: return [], []

    embs = encode_sentences_sbert(sentences)
    n = embs.shape[0]
    coherence = [0.0] * n
    semantic_shift = [0.0] * n

    for t in range(1, n):
        # Local Coherence: Similarity with the immediate predecessor
        coherence[t] = round(float(np.dot(embs[t-1], embs[t])), 4)

        # Semantic Shift: Distance from the running average of previous context
        mean_prev = embs[:t].mean(axis=0)
        mean_prev /= (np.linalg.norm(mean_prev) + 1e-9) # Re-normalize mean vector
        
        cos_sim = float(np.dot(embs[t], mean_prev))
        semantic_shift[t] = round(1.0 - cos_sim, 4)

    return coherence, semantic_shift

# ==========================================
# 4. Main Integration Pipeline
# ==========================================
def main():
    base_path = "/home/qgroup2/sungpil/flash_fiction"
    input_file = os.path.join(base_path, "flash_fiction_with_surprisal.csv")
    output_file = os.path.join(base_path, "flash_fiction_with_surprisal_coherence_semantic.csv")

    if not os.path.exists(input_file):
        print("Error: Input file (Surprisal CSV) not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Processing {len(df)} stories for SBERT metrics...")

    all_sentence_lists = []
    all_coherence_vectors = []
    all_semantic_shift_vectors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Discourse"):
        try:
            sentences = split_sentences_combined(row['text'], tokenizer)
            coh_vec, sem_vec = calculate_coherence_and_semantic_shift(sentences)

            all_sentence_lists.append(str(sentences))
            all_coherence_vectors.append(str(coh_vec))
            all_semantic_shift_vectors.append(str(sem_vec))
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            for l in [all_sentence_lists, all_coherence_vectors, all_semantic_shift_vectors]:
                l.append("[]")

    # Update DataFrame and export
    df['sentence_list'] = all_sentence_lists
    df['coherence_vector'] = all_coherence_vectors
    df['semantic_shift_vector'] = all_semantic_shift_vectors

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully integrated SBERT metrics. Output: {output_file}")

if __name__ == "__main__":
    main()