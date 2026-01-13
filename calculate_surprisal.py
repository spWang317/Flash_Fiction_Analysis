import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from konlpy.tag import Mecab
import kss

# ==========================================
# 1. Advanced Sentence Splitting Functions
# ==========================================
try:
    mecab = Mecab()
except Exception:
    from konlpy.tag import Mecab
    mecab = Mecab()

def clean_sentences(sentences):
    """Removes noise: short strings or those consisting only of symbols."""
    cleaned = []
    for s in sentences:
        stripped = s.strip()
        if len(stripped) <= 2 and all(c in '.,!?“”‘’"\' ' for c in stripped):
            continue
        cleaned.append(stripped)
    return cleaned

def split_sentences_improved(text):
    """Fallback Mecab-based splitter for complex Korean narratives."""
    try:
        tokens = mecab.pos(text)
    except:
        return [text]

    boundaries = []
    i = 0
    maj_set = {'그러나', '그런데', '그리고', '그래서', '그래도', '따라서', '하지만', '결국'}
    while i < len(tokens):
        word, pos = tokens[i]
        if pos in {'EF', 'SF'}: # Sentence endings
            boundaries.append(i)
        elif pos == 'MAJ' and word in maj_set: # Conjunctions
            boundaries.append(i - 1 if i > 0 else 0)
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
    """Segments sentences exceeding the token limit to ensure model stability."""
    input_ids = tokenizer.encode(sent, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return [sent]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text: chunks.append(chunk_text)
    return chunks

def split_sentences_combined(text, tokenizer):
    """Main pipeline: KSS -> Mecab (Fallback) -> Token-based Sub-splitting."""
    if not isinstance(text, str) or not text.strip():
        return []
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
# 2. Model Loading (Solar-10.7B)
# ==========================================
MODEL_ID = "beomi/OPEN-SOLAR-KO-10.7B"

# 4-bit Quantization configuration for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading Model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# Solar context window size management
MAX_CONTEXT = 3500 

# ==========================================
# 3. Surprisal Calculation Implementation
# ==========================================
def calculate_surprisal_vector(text):
    """Calculates sentence-level surprisals (Negative Log-Likelihood)."""
    sentences = split_sentences_combined(text, tokenizer)
    if not sentences: return []

    surprisals = []
    history_ids = []

    with torch.no_grad():
        for i, sent in enumerate(sentences):
            add_special_tokens = (i == 0)
            sent_ids = tokenizer.encode(sent, add_special_tokens=add_special_tokens)

            # Build full input with context
            input_ids = history_ids + sent_ids
            input_tensor = torch.tensor([input_ids]).to(model.device)

            # Target mask: focus loss calculation only on the current sentence (-100 ignored)
            target_ids = [-100] * len(history_ids) + sent_ids
            target_tensor = torch.tensor([target_ids]).to(model.device)

            outputs = model(input_tensor, labels=target_tensor)
            nll = outputs.loss.item()

            if not (np.isnan(nll) or np.isinf(nll)):
                surprisals.append(round(nll, 4))

            # Maintain history and enforce sliding window
            history_ids.extend(sent_ids)
            if len(history_ids) > MAX_CONTEXT:
                history_ids = history_ids[-MAX_CONTEXT:]

    return surprisals

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    base_path = "/home/qgroup2/sungpil/short_novel"
    input_file = os.path.join(base_path, "short_novel_merged_filtered.csv")
    output_file = os.path.join(base_path, "short_novel_with_surprisal.csv")
    
    if not os.path.exists(input_file):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Total rows to process: {len(df)}")
    
    all_surprisal_vectors = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Surprisal"):
        try:
            vector = calculate_surprisal_vector(row['text'])
            all_surprisal_vectors.append(str(vector))
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            all_surprisal_vectors.append("[]")
            
    df['surprisal_vector'] = all_surprisal_vectors
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Processing Complete. Saved to: {output_file}")

if __name__ == "__main__":
    main()