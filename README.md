# 📚 Computational Analysis of Flash Fiction

This research investigates the narrative dynamics of Korean flash fiction through computational methods. The pipeline is designed to transform raw narrative texts into multi-dimensional signal trajectories (Surprisal, Coherence, and Semantic Shift) for structural analysis.

---

## 🛠 Installation Guide (Setup for Flash Fiction Analysis)

This guide provides instructions for setting up the environment for the Korean Flash Fiction analysis project. The project includes features for sentence-level statistical extraction, LLM-based Surprisal calculation, and SBERT-based discourse analysis.

### 1. Environment Setup

#### Method A: Conda (Recommended)
Create the `flashfiction_analysis` environment using the `environment.yml` file.
```bash
conda env create -f environment.yml
conda activate flashfiction_analysis
```

#### Method B: Pip
Manually create the environment and install dependencies via `requirements.txt`.
```bash
conda create -n flashfiction_analysis python=3.10 -y
conda activate flashfiction_analysis
pip install -r requirements.txt
```

---

### 2. Mecab Installation (OS-specific Setup)

The `Mecab` engine is required for Korean morphological analysis.

#### 🍎 macOS (Apple Silicon/Intel)
**Case 1: Homebrew is installed**
```bash
# Unlink to prevent conflicts between generic mecab and mecab-ko
brew unlink mecab
brew install mecab-ko mecab-ko-dic
brew link mecab-ko
pip install mecab-python3
```

**Case 2: Homebrew is NOT installed**
```bash
bash <(curl -s [https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh](https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh))
pip install mecab-python3
```

#### 🪟 Windows
1. **Java JDK Installation**: Install from the Oracle website and set the `JAVA_HOME` environment variable.
2. **Mecab Binary**: Download the binary from [mecab-ko-msvc](https://github.com/PHeonix-P/mecab-ko-msvc/releases) and extract it to `C:\mecab`.
3. **Python Library**: 
   ```bash
   pip install mecab-python3
   ```

#### 🐧 Linux (GPU Server)
Use the official script to install the Mecab engine directly on the system.
```bash
bash <(curl -s [https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh](https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh))
```

---

### 3. KSS and Common Libraries
The project utilizes `KSS` for Korean sentence segmentation.
```bash
pip install kss
```
*Note: Windows users may need to install 'Visual Studio Build Tools' if C++ build errors occur during installation.*

---

### 4. Verification
Verify the installation by running the following Python code to ensure all modules load correctly.

```python
from konlpy.tag import Mecab
import kss

# 1. Mecab Morphological Analysis Test
try:
    mecab = Mecab()
    print("Mecab Morph Test:", mecab.morphs("디지털 인문학 분석을 시작합니다."))
except Exception as e:
    print("Mecab Error: Check if the mecab engine is installed correctly.")

# 2. KSS Sentence Segmentation Test
text = "안녕하세요. 문장 분리 테스트 중입니다. 잘 작동하나요?"
print("KSS Split Test:", kss.split_sentences(text))
```

---

## ⚠️ Important Note for Server Users
* **GPU Acceleration**: `calculate_surprisal.py` utilizes 4-bit quantization via `BitsAndBytesConfig`, which requires a **Linux environment and an NVIDIA GPU (CUDA)**.
* **SBERT Model**: The discourse analysis script (`coherence_topic_calc.py`) uses the `jhgan/ko-sroberta-multitask` model to generate sentence embeddings.
* **Execution Recommendation**: Use local environments (Mac/Windows) for statistical analysis and visualization (`check_sent_stats.py`) and use a GPU server for heavy LLM computations.

---

## 📂 Data Inventory

> **Note on Copyright:** Due to copyright restrictions, full narrative texts and segmented sentence lists are not provided in the public repository. Metadata summaries are provided instead.

### 1. Metadata: `book_list_summary.csv`
Contains bibliographic information of the source texts.
* **isbn**: International Standard Book Number (Primary Key)
* **book_title / author / publisher / pub_year / country**

### 2. Analysis Files
* `flash_fiction_merged.csv`: Initial raw dataset with full texts (Not provided).
* `flash_fiction_merged_filtered.csv`: Refined dataset after outlier removal.
* `flash_fiction_with_surprisal_coherence_semantic.csv`: **[Final Master File]** Contains all integrated narrative signal vectors for final analysis.

---

## ⚙️ Step-by-Step Pipeline

The narrative trajectories are generated through a three-stage sequence:

### **[Step 1] Preprocessing & Filtering: `check_sent_stats.py`**
* **Input:** `flash_fiction_merged.csv`
* **Process:** * Sentence segmentation using `KSS` and `Mecab`.
    * Statistical analysis of sentence, character, and token counts.
    * **Outlier Removal:** Stories in the bottom 5% and top 5% of the sentence count distribution are excluded to ensure homogeneity.
* **Output:** `flash_fiction_merged_filtered.csv`, `length_analysis.zip`

### **[Step 2] Surprisal Extraction: `calculate_surprisal.py`**
* **Input:** `flash_fiction_merged_filtered.csv`
* **Model:** `Solar-10.7B` (LLM) utilizing a 3,500 token sliding window.
* **Metric:** Sentence-level Surprisal (Negative Log-Likelihood).
* **Logic:** Measures the information-theoretic "shock" of each sentence within its narrative context.
* **Output:** `flash_fiction_with_surprisal.csv`

### **[Step 3] Discourse Signal Calculation: `coherence_topic_calc.py`**
* **Input:** `flash_fiction_with_surprisal.csv`
* **Model:** `ko-sroberta-multitask` (SBERT)
* **Metrics:**
    * **Local Coherence:** Cosine similarity between adjacent sentence embeddings.
    * **Semantic Shift:** Deviation of the current sentence from the cumulative context mean.
* **Output:** `flash_fiction_with_surprisal_coherence_semantic.csv` 

---

## 🧪 Final Analysis

The master output is utilized in **`FlashFiction_Analysis.ipynb`**. This notebook performs:
* **Input:** `flash_fiction_with_surprisal_coherence_semantic.csv` 
1. **Stability Diagnostics**: Detection and removal of initial 'burn-in' noise.
2. **Trajectory Clustering**: Identification of narrative archetypes and structural patterns.
3. **Peak Dynamics**: Point-wise and dynamic recovery analysis (TTR, Slope) following narrative shocks.

**Upon execution, all generated statistical reports and numerical summaries are saved to the `statistical_outputs/` directory, while all visualization plots and figures are exported to the `figure_outputs/` directory for further review.**



## 📄 Citation & Preprint

If you find this framework or the provided semantic dynamics analysis useful for your research, please cite our preprint:
**"The Fracture and Leap Cycle: Quantifying Narrative Surprise and Structural Resilience in Flash Fiction"**

* **Preprint Server**: Research Square
* **DOI**: [10.21203/rs.3.rs-8619161/v1](https://doi.org/10.21203/rs.3.rs-8619161/v1)
* **URL**: [https://www.researchsquare.com/article/rs-8619161/v1](https://www.researchsquare.com/article/rs-8619161/v1)
