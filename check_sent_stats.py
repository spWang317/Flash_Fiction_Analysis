import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import kss
from konlpy.tag import Mecab

# Set Matplotlib backend for terminal environments (non-GUI)
plt.switch_backend('agg')

# ==========================================
# Initialize Morphological Analyzer
# ==========================================
try:
    mecab = Mecab()
except Exception:
    from konlpy.tag import Mecab
    mecab = Mecab()

# ==========================================
# Sentence Splitting Utilities
# ==========================================
def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        stripped = s.strip()
        # Remove segments with length <= 2 consisting only of punctuation or whitespace
        if len(stripped) <= 2 and all(c in '.,!?“”‘’"\' ' for c in stripped):
            continue
        cleaned.append(stripped)
    return cleaned

def split_sentences_improved(text):
    try:
        tokens = mecab.pos(text)
    except Exception:
        return [text]

    boundaries = []
    i = 0
    maj_set = {'그러나', '그런데', '그리고', '그래서', '그래도', '따라서', '하지만', '결국'}
    vv_ec_phrases = {('그러', '자'), ('그러', '면'), ('하', '니까'), ('하', '여서'), ('그리', '하여')}

    while i < len(tokens):
        word, pos = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ('', '')
        next_word, next_pos = next_token

        if pos in {'EF', 'SF'}:
            boundaries.append(i)
        elif pos == 'EC' and next_pos == 'MAJ':
            boundaries.append(i)
        elif pos == 'MAJ' and word in maj_set:
            boundaries.append(i - 1 if i > 0 else 0)
        elif pos == 'VV' and (word, next_word) in vv_ec_phrases and next_pos == 'EC':
            boundaries.append(i + 1)
            i += 1
        elif pos == 'EF' and i + 2 < len(tokens):
            nn_word, nn_pos = tokens[i + 2]
            if next_pos in {'NP'} or next_word in {'나', '너', '그', '그녀', '당신'}:
                if nn_pos == 'JKS':
                    boundaries.append(i)
        elif pos == 'EF' and next_pos in {'VV', 'VA'}:
            boundaries.append(i)
        i += 1

    result = []
    start_idx = 0
    words = [w for w, _ in tokens]
    offsets = []
    idx = 0
    text_len = len(text)
    
    for w in words:
        while idx < text_len and text[idx].isspace():
            idx += 1
        start = idx
        idx += len(w)
        offsets.append((start, idx))

    if tokens:
        boundaries.append(len(tokens) - 1)

    for b in boundaries:
        if b < len(offsets):
            end = offsets[b][1]
            sentence = text[start_idx:end].strip()
            if sentence:
                result.append(sentence)
            start_idx = end
            
    if start_idx < text_len:
        remain = text[start_idx:].strip()
        if remain:
            result.append(remain)
    return result

def split_long_sentence_by_tokens(sent, tokenizer, max_tokens=256):
    input_ids = tokenizer.encode(sent, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return [sent]
    
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

def split_sentences_combined(text, tokenizer):
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        kss_split = kss.split_sentences(text)
    except Exception:
        kss_split = split_sentences_improved(text)

    kss_split = clean_sentences(kss_split)

    final_sents = []
    for s in kss_split:
        final_sents.extend(split_long_sentence_by_tokens(s, tokenizer))
    return final_sents

# ==========================================
# Main Execution Logic
# ==========================================
def main():
    print(">>> Program Started!") 
    
    # Input/Output paths
    base_path = "/home/qgroup2/sungpil/short_novel"
    input_file = os.path.join(base_path, "short_novel_merged.csv")
    
    MODEL_ID = "beomi/OPEN-SOLAR-KO-10.7B"
    print(f"Loading Tokenizer: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Reading CSV from: {input_file}")
    if not os.path.exists(input_file):
        print("Error: Input file not found.")
        return
    df = pd.read_csv(input_file)
    print(f"Total rows: {len(df)}")
    
    # Handle NaN values
    df['text'] = df['text'].fillna("")

    real_sent_counts = []
    char_counts = []
    token_counts = []

    print("Calculating sentence/char/token counts...")

    for text in tqdm(df['text']):
        try:
            sents = split_sentences_combined(text, tokenizer)
            n_sent = len(sents)
            n_char = len(text)
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            n_sent = 0
            n_char = 0
            n_tok = 0

        real_sent_counts.append(n_sent)
        char_counts.append(n_char)
        token_counts.append(n_tok)

    # --------------------------------------
    # Descriptive Statistics Output
    # --------------------------------------
    print("\n" + "="*40)
    print("[Sentence Count Statistics]")
    print(f"Mean: {np.mean(real_sent_counts):.2f}")
    print(f"Median: {np.median(real_sent_counts)}")
    print(f"Min: {np.min(real_sent_counts)}")
    print(f"Max: {np.max(real_sent_counts)}")
    print(f"10th Percentile: {np.percentile(real_sent_counts, 10):.2f}")
    print(f"25th Percentile: {np.percentile(real_sent_counts, 25):.2f}")
    print(f"75th Percentile: {np.percentile(real_sent_counts, 75):.2f}")
    print(f"90th Percentile: {np.percentile(real_sent_counts, 90):.2f}")
    print("="*40)

    # --------------------------------------
    # Save Per-Story CSV
    # (Adjust column names based on the actual schema, e.g., 'book_id', 'story_id')
    # --------------------------------------
    df['n_sentences'] = real_sent_counts
    df['n_chars'] = char_counts
    df['n_tokens'] = token_counts

    length_path = os.path.join(base_path, "length_stats_per_story.csv")
    # If metadata columns exist, select and save them along with counts
    cols_to_save = ['n_sentences', 'n_chars', 'n_tokens']
    for c in ['book_id', 'story_id', 'title']:
        if c in df.columns:
            cols_to_save.insert(0, c)
    df[cols_to_save].to_csv(length_path, index=False)
    print(f"Per-story length stats saved to: {length_path}")

    # --------------------------------------
    # Save Summary Statistics CSV
    # --------------------------------------
    summary = {
        "mean_sent": float(np.mean(real_sent_counts)),
        "median_sent": float(np.median(real_sent_counts)),
        "min_sent": int(np.min(real_sent_counts)),
        "max_sent": int(np.max(real_sent_counts)),
        "p10_sent": float(np.percentile(real_sent_counts, 10)),
        "p25_sent": float(np.percentile(real_sent_counts, 25)),
        "p75_sent": float(np.percentile(real_sent_counts, 75)),
        "p90_sent": float(np.percentile(real_sent_counts, 90)),
        "mean_chars": float(np.mean(char_counts)),
        "median_chars": float(np.median(char_counts)),
        "p25_chars": float(np.percentile(char_counts, 25)),
        "p75_chars": float(np.percentile(char_counts, 75))
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(base_path, "length_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary stats saved to: {summary_path}")

    # --------------------------------------
    # Save Figures
    # --------------------------------------
    # Sentence Count Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(real_sent_counts, bins=100, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(np.median(real_sent_counts), color='yellow', linestyle='-', linewidth=2, label='Median')
    plt.title("Distribution of Sentence Counts per Story")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Count")
    plt.legend()
    sent_hist_path = os.path.join(base_path, "sentence_dist_check.png")
    plt.savefig(sent_hist_path)
    print(f"Sentence histogram saved to: {sent_hist_path}")

    # Character Count Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(char_counts, bins=100, color='slateblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Character Counts per Story")
    plt.xlabel("Number of Characters")
    plt.ylabel("Count")
    char_hist_path = os.path.join(base_path, "char_dist_check.png")
    plt.savefig(char_hist_path)
    print(f"Char histogram saved to: {char_hist_path}")

    # Sentence vs Character Counts Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(real_sent_counts, char_counts, alpha=0.3, s=5)
    plt.xlabel("Number of Sentences")
    plt.ylabel("Number of Characters")
    plt.title("Sentence vs Character Counts per Story")
    scatter_path = os.path.join(base_path, "sent_vs_char_scatter.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to: {scatter_path}")

    print("\n>>> Done.")

if __name__ == "__main__":
    main()