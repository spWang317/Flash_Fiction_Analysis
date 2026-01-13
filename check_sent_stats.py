import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
import kss
from konlpy.tag import Mecab
import shutil

# Set Matplotlib backend for server environments
plt.switch_backend('agg')

# ==========================================
# 0. Tool Initialization
# ==========================================
try:
    mecab = Mecab()
except Exception:
    from konlpy.tag import Mecab
    mecab = Mecab()

def clean_sentences(sentences):
    """Removes empty or punctuation-only strings from the sentence list."""
    cleaned = []
    for s in sentences:
        stripped = s.strip()
        if len(stripped) <= 2 and all(c in '.,!?“”‘’"\' ' for c in stripped):
            continue
        cleaned.append(stripped)
    return cleaned

def split_sentences_improved(text):
    """Fallback sentence splitter using Mecab POS tagging."""
    try:
        tokens = mecab.pos(text)
    except:
        return [text]
    boundaries = []
    i = 0
    while i < len(tokens):
        word, pos = tokens[i]
        if pos in {'EF', 'SF'}: boundaries.append(i)
        i += 1
    result = []
    start_idx = 0
    text_len = len(text)
    words = [w for w, _ in tokens]
    offsets = []
    idx = 0
    for w in words:
        while idx < text_len and text[idx].isspace(): idx += 1
        start = idx; idx += len(w); offsets.append((start, idx))
    if tokens: boundaries.append(len(tokens) - 1)
    for b in boundaries:
        if b < len(offsets):
            end = offsets[b][1]
            sentence = text[start_idx:end].strip()
            if sentence: result.append(sentence)
            start_idx = end
    return result

def split_long_sentence_by_tokens(sent, tokenizer, max_tokens=256):
    """Sub-segments sentences that exceed the maximum token limit."""
    input_ids = tokenizer.encode(sent, add_special_tokens=False)
    if len(input_ids) <= max_tokens: return [sent]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text: chunks.append(chunk_text)
    return chunks

def split_sentences_combined(text, tokenizer):
    """Main sentence splitting pipeline combining KSS and token-based constraints."""
    if not isinstance(text, str) or not text.strip(): return []
    try:
        kss_split = kss.split_sentences(text)
    except:
        kss_split = split_sentences_improved(text)
    kss_split = clean_sentences(kss_split)
    final_sents = []
    for s in kss_split:
        final_sents.extend(split_long_sentence_by_tokens(s, tokenizer))
    return final_sents

# ==========================================
# 1. Visualization and Statistics Export
# ==========================================
def save_results(df, suffix, output_dir, lower_limit=None, upper_limit=None):
    """Generates distribution plots and summary CSVs."""
    print(f"\n>>> Saving [{suffix.upper()}] results...")
    
    s_counts = df['n_sentences'].values
    c_counts = df['n_chars'].values
    t_counts = df['n_tokens'].values

    # Plot font size configuration
    TITLE_SIZE = 30
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24

    # 1) Histogram of Sentence Counts
    plt.figure(figsize=(12, 8))
    plt.hist(s_counts, bins=70, color='teal', alpha=0.7, edgecolor='black')
    
    # Median line
    median_val = np.median(s_counts)
    plt.axvline(median_val, color='orange', linestyle='-', linewidth=4, label=f'Median: {median_val}')
    
    # Threshold lines for Raw data
    if suffix == 'raw' and lower_limit is not None and upper_limit is not None:
        plt.axvline(lower_limit, color='red', linestyle='--', linewidth=4, label=f'5% Lower ({lower_limit:.1f})')
        plt.axvline(upper_limit, color='red', linestyle='--', linewidth=4, label=f'95% Upper ({upper_limit:.1f})')
    
    plt.title(f"Sentence Count Distribution ({suffix.upper()})", fontsize=TITLE_SIZE, pad=20)
    plt.xlabel("Number of Sentences", fontsize=LABEL_SIZE)
    plt.ylabel("Frequency", fontsize=LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE, loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"03_dist_plot_{suffix}.png"), dpi=200)
    plt.close()

    # 2) Scatter Plot with Regression line
    plt.figure(figsize=(10, 8))
    plt.scatter(s_counts, t_counts, alpha=0.3, s=20, color='darkgreen')
    
    if suffix == 'filtered':
        m, b = np.polyfit(s_counts, t_counts, 1) # Linear regression
        plt.plot(s_counts, m*s_counts + b, color='red', linewidth=4, label=f'y = {m:.2f}x + {b:.2f}')
        plt.legend(fontsize=LEGEND_SIZE)

    plt.title(f"Sentence vs Token Correlation ({suffix.upper()})", fontsize=TITLE_SIZE, pad=20)
    plt.xlabel("Number of Sentences", fontsize=LABEL_SIZE)
    plt.ylabel("Number of Tokens", fontsize=LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"04_scatter_plot_{suffix}.png"), dpi=200)
    plt.close()

    # 3) Summary Statistics CSV
    summary_path = os.path.join(output_dir, f"02_summary_{suffix}.csv")
    summary_df = pd.DataFrame({
        "Measure": ["Sentences", "Characters", "Tokens"],
        "Min": [np.min(s_counts), np.min(c_counts), np.min(t_counts)],
        "Max": [np.max(s_counts), np.max(c_counts), np.max(t_counts)],
        "Mean": [np.mean(s_counts), np.mean(c_counts), np.mean(t_counts)],
        "Median": [np.median(s_counts), np.median(c_counts), np.median(t_counts)],
        "IQR": [f"{np.percentile(s_counts, 25)}-{np.percentile(s_counts, 75)}",
                f"{np.percentile(c_counts, 25)}-{np.percentile(c_counts, 75)}",
                f"{np.percentile(t_counts, 25)}-{np.percentile(t_counts, 75)}"]
    })
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

# ==========================================
# 2. Main Execution Pipeline
# ==========================================
def main():
    base_path = "/home/qgroup2/sungpil/short_novel"
    output_dir = os.path.join(base_path, "length_analysis")
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(base_path, "short_novel_merged.csv")
    # Standalone filtered file path (Outside the ZIP)
    filtered_output_file = os.path.join(base_path, "short_novel_merged_filtered.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("beomi/OPEN-SOLAR-KO-10.7B")

    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file).fillna("")

    real_sent_counts, char_counts, token_counts = [], [], []
    for text in tqdm(df['text'], desc="Calculating Length Metrics"):
        try:
            sents = split_sentences_combined(text, tokenizer)
            n_sent, n_char = len(sents), len(text)
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        except:
            n_sent, n_char, n_tok = 0, 0, 0
        real_sent_counts.append(n_sent); char_counts.append(n_char); token_counts.append(n_tok)

    df['n_sentences'], df['n_chars'], df['n_tokens'] = real_sent_counts, char_counts, token_counts

    # Calculate 5th and 95th quantiles for filtering
    lower = df['n_sentences'].quantile(0.05)
    upper = df['n_sentences'].quantile(0.95)

    # 1. Save Raw analysis (Distribution including outlier lines)
    save_results(df, "raw", output_dir, lower_limit=lower, upper_limit=upper)

    # 2. Filtering Outliers
    df_filtered = df[(df['n_sentences'] >= lower) & (df['n_sentences'] <= upper)].copy()
    
    # 3. Save Filtered analysis results (Figures and Summary inside the folder)
    save_results(df_filtered, "filtered", output_dir)

    # 4. Export the Filtered Standalone CSV (Requested: Outside the ZIP)
    df_filtered.to_csv(filtered_output_file, index=False, encoding="utf-8-sig")
    print(f"\n>>> Filtered standalone CSV saved: {filtered_output_file}")

    # Zip the analysis plots and summary tables
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f">>> Analysis archive created: {output_dir}.zip")

if __name__ == "__main__":
    main()