# 📚 Computational Analysis of Flash Fiction

This repository provides the full computational pipeline and pre-computed signals for the study:
**"The Fracture and Leap Cycle: Quantifying Narrative Surprise and Structural Resilience in Flash Fiction"**

> **Note on Copyright:** Due to copyright restrictions, the original narrative texts cannot be shared publicly. However, the pre-computed numerical signals are provided, allowing immediate reproduction of all statistical analyses and figures reported in the paper.

---

## ⚡ Quick Start (Recommended)

All statistical analyses and figures in the paper can be immediately reproduced using the pre-computed master dataset.

### Requirements
- Python 3.10+
- Jupyter Notebook

### Steps

1. Install dependencies:

```bash
conda env create -f environment.yml
conda activate flashfiction_analysis
```

or

```bash
pip install -r requirements.txt
```

2. Open and run the notebook:

```bash
jupyter notebook FlashFiction_Analysis.ipynb
```

The notebook will automatically load `flash_fiction_with_surprisal_coherence_semantic.csv` and reproduce all results. Statistical outputs are saved to `statistical_outputs/` and figures to `figure_outputs/`.

---

## 📂 Data Inventory

| File | Available | Description |
|------|-----------|-------------|
| `flash_fiction_with_surprisal_coherence_semantic.csv` | ✅ | Master dataset with pre-computed Surprisal, Coherence, and Semantic Shift signals |
| `book_list_summary.csv` | ✅ | Bibliographic metadata for corpus reconstruction |
| `flash_fiction_merged.csv` | ❌ | Raw dataset with full texts (copyright restricted) |
| `flash_fiction_merged_filtered.csv` | ❌ | Filtered dataset (copyright restricted) |

### `book_list_summary.csv` columns
- **isbn**: International Standard Book Number (Primary Key)
- **book_title / author / publisher / pub_year / country**

Researchers wishing to reconstruct the corpus may obtain the original texts via commercially available editions or library services using the metadata provided.

---

## ⚙️ Full Pipeline (For researchers with original texts)

If you have access to the original narrative texts, the full three-stage pipeline can be executed as follows.

### Environment Setup

#### Method A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate flashfiction_analysis
```

#### Method B: Pip

```bash
conda create -n flashfiction_analysis python=3.10 -y
conda activate flashfiction_analysis
pip install -r requirements.txt
```

### Mecab Installation (Required for Step 1)

#### 🍎 macOS

```bash
brew unlink mecab
brew install mecab-ko mecab-ko-dic
brew link mecab-ko
pip install mecab-python3
```

#### 🪟 Windows

1. Install Java JDK and set `JAVA_HOME`
2. Download [mecab-ko-msvc](https://github.com/PHeonix-P/mecab-ko-msvc/releases) and extract to `C:\mecab`
3. `pip install mecab-python3`

#### 🐧 Linux

```bash
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

### KSS Installation

```bash
pip install kss
```

### Installation Verification

```python
from konlpy.tag import Mecab
import kss

try:
    mecab = Mecab()
    print("Mecab Morph Test:", mecab.morphs("디지털 인문학 분석을 시작합니다."))
except Exception as e:
    print("Mecab Error: Check if the mecab engine is installed correctly.")

text = "안녕하세요. 문장 분리 테스트 중입니다. 잘 작동하나요?"
print("KSS Split Test:", kss.split_sentences(text))
```

---

### Step 1. Preprocessing & Filtering: `check_sent_stats.py`

- **Input:** `flash_fiction_merged.csv` (original texts required)
- **Process:** Sentence segmentation (KSS + Mecab), outlier removal (bottom/top 5% by sentence count)
- **Output:** `flash_fiction_merged_filtered.csv`, `length_analysis.zip`

### Step 2. Surprisal Extraction: `calculate_surprisal.py`

- **Input:** `flash_fiction_merged_filtered.csv`
- **Model:** `beomi/OPEN-SOLAR-KO-10.7B` with 3500-token sliding window
- **Requirement:** Linux + NVIDIA GPU (CUDA) for 4-bit quantization
- **Output:** `flash_fiction_with_surprisal.csv`

### Step 3. Discourse Signal Calculation: `coherence_topic_calc.py`

- **Input:** `flash_fiction_with_surprisal.csv`
- **Model:** `jhgan/ko-sroberta-multitask`
- **Metrics:** Local Coherence (cosine similarity between adjacent sentences), Semantic Shift (deviation from cumulative context mean)
- **Output:** `flash_fiction_with_surprisal_coherence_semantic.csv`

---

## 📄 Citation

If you find this repository useful, please cite:

**Wang, S. (2026).** The Fracture and Leap Cycle: Quantifying Narrative Surprise and Structural Resilience in Flash Fiction.

- **Preprint DOI:** [10.21203/rs.3.rs-8619161/v1](https://doi.org/10.21203/rs.3.rs-8619161/v1)
- **Repository DOI:** [10.5281/zenodo.19625501](https://doi.org/10.5281/zenodo.19625501)
