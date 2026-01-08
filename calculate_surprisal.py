import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from konlpy.tag import Mecab
import kss
from collections import Counter

# ==========================================
# 1. Sentence Split Functions
# ==========================================
try:
    mecab = Mecab()
except Exception as e:
    print("Warning: Mecab not found via konlpy directly. Trying fallback.")
    from konlpy.tag import Mecab
    mecab = Mecab()

def clean_sentences(sentences):
    """
    Remove noise sentences that are too short or consist only of symbols.
    """
    cleaned = []
    for s in sentences:
        stripped = s.strip()
        # Remove if length is <= 2 and consists only of punctuation/whitespace (e.g., ".", "?", " ")
        if len(stripped) <= 2 and all(c in '.,!?“”‘’"\' ' for c in stripped):
            continue
        cleaned.append(stripped)
    return cleaned

def split_sentences_improved(text):
    """
    [Fallback] Mecab-based sentence splitter used only when KSS fails.
    Splits conservatively based on final endings (EF, SF).
    """
    try:
        tokens = mecab.pos(text)
    except Exception:
        return [text] # Return as a single chunk if Mecab also fails

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
    
    # Map index to original text to prevent issues during detokenization
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
            
    # Handle remaining text
    if start_idx < text_len:
        remain = text[start_idx:].strip()
        if remain:
            result.append(remain)
            
    return result

# Sub-splitting based on token length for extremely long sentences.
# Considering Solar model's training distribution, 256-512 tokens is an appropriate chunk size.
MAX_SENT_TOKENS = 256  

def split_long_sentence_by_tokens(sent, tokenizer, max_tokens=MAX_SENT_TOKENS):
    """
    Tokenizes a single sentence and if the token count exceeds max_tokens,
    creates chunks and decodes them back into a list of strings.
    (Safety measure for texts like 'stream of consciousness' with no periods)
    """
    input_ids = tokenizer.encode(sent, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return [sent]
    
    chunks = []
    # Simple slicing without overlap
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

def split_sentences_combined(text, tokenizer):
    """
    [Final Sentence Splitting Function]
    1. KSS (Primary): Grammatical sentence splitting
    2. Mecab (Fallback): Rule-based splitting if KSS fails
    3. Token Split (Safety): Prevents exceeding model context length
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1) Primary: Use KSS
    try:
        kss_split = kss.split_sentences(text)
    except Exception:
        # 2) Fallback: Mecab-based backup
        kss_split = split_sentences_improved(text)

    # 3) Remove short sentences/symbols
    kss_split = clean_sentences(kss_split)

    # 4) Post-split each sentence based on token length
    final_sents = []
    for s in kss_split:
        final_sents.extend(split_long_sentence_by_tokens(s, tokenizer))
        
    return final_sents

# ==========================================
# 2. Model Loading (Solar-10.7B)
# ==========================================
MODEL_ID = "beomi/OPEN-SOLAR-KO-10.7B"

print(f"Loading Model: {MODEL_ID}...")

# 4-bit Quantization Config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()  # Set to Evaluation Mode

print("Model Loaded Successfully.")

# Solar context window size (including safety buffer)
MAX_CONTEXT = 3500 

# ==========================================
# 3. Surprisal Calculation Logic
# ==========================================
def calculate_surprisal_vector(text):
    sentences = split_sentences_combined(text, tokenizer)
    if not sentences:
        return []

    surprisals = []
    history_ids = []

    with torch.no_grad():
        for i, sent in enumerate(sentences):
            add_special_tokens = (i == 0)
            sent_ids = tokenizer.encode(sent, add_special_tokens=add_special_tokens)

            input_ids = history_ids + sent_ids
            input_tensor = torch.tensor([input_ids]).to(model.device)

            target_ids = [-100] * len(history_ids) + sent_ids
            target_tensor = torch.tensor([target_ids]).to(model.device)

            outputs = model(input_tensor, labels=target_tensor)
            nll = outputs.loss.item()

            # NaN/Inf Defense: Skip surprisal for this sentence but keep history
            if np.isnan(nll) or np.isinf(nll):
                history_ids.extend(sent_ids)
                if len(history_ids) > MAX_CONTEXT:
                    history_ids = history_ids[-MAX_CONTEXT:]
                continue

            surprisals.append(round(nll, 4))

            history_ids.extend(sent_ids)
            if len(history_ids) > MAX_CONTEXT:
                history_ids = history_ids[-MAX_CONTEXT:]

    return surprisals


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    base_path = "/home/qgroup2/sungpil/short_novel"
    input_file = os.path.join(base_path, "short_novel_merged.csv")
    output_file = os.path.join(base_path, "short_novel_with_surprisal.csv")
    
    print(f"Reading CSV from: {input_file}")
    
    if not os.path.exists(input_file):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_file)
    
    # Check for 'text' column
    if 'text' not in df.columns:
        print("Error: CSV must contain a 'text' column.")
        return
        
    print(f"Total rows to process: {len(df)}")
    
    # List to store results
    all_surprisal_vectors = []
    
    # Process with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Novels"):
        try:
            text = row['text']
            vector = calculate_surprisal_vector(text)
            all_surprisal_vectors.append(str(vector))  # Convert to string for CSV storage
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            all_surprisal_vectors.append("[]")
            
    # Add new column
    df['surprisal_vector'] = all_surprisal_vectors
    
    # Save output
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("=" * 40)
    print(f"Processing Complete!")
    print(f"Saved to: {output_file}")
    print("=" * 40)

if __name__ == "__main__":
    main()