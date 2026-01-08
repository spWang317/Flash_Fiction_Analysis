# Korean Flash Fiction Narrative Analysis

This project provides a computational framework to analyze the narrative structures of Korean short stories (flash fiction). It utilizes Large Language Models (LLMs) to calculate **Surprisal**, and Sentence-BERT (SBERT) to measure **Local Coherence** and **Global Semantic Shift**.

---

## 1. Technical Specifications

The following hardware and software environment was used for neural inference and statistical analysis to ensure high-performance computing (HPC) stability.

* **OS**: Ubuntu 24.04.2 LTS
* **CPU**: Intel(R) Xeon(R) Silver 4516Y+
* **RAM**: 251 GiB
* **GPU**: Dual NVIDIA RTX A6000 (48 GB VRAM each, **Total 96 GB VRAM**)
* **Architecture**: Optimized for intensive transformer-based inference and large-scale corpus processing.

---

## 2. Data Inventory & Policy

Before setting up the environment, please ensure you understand the data structure required for this research.

### (1) Input Dataset: `short_novel_merged.csv` (Not Provided)
Due to **copyright restrictions** regarding proprietary literary content, the full-text dataset is **not publicly provided**. To run the analysis, you must prepare your own CSV file with the following schema:

| Column Name | Description |
|:---|:---|
| `isbn` | International Standard Book Number |
| `story_title` | Title of the individual flash fiction story |
| `text` | **Full narrative text** (Required for processing) |
| `author` | Name of the author |
| `birth_year` | Birth year of the author |
| `gender` | Gender of the author |
| `book_title` | Title of the source book or anthology |
| `publisher` | Name of the publisher |
| `pub_year` | Year of publication |
| `country` | Country of author |

### (2) Reference Data: `book_list_summary.csv` (Provided)
We provide a **bibliographic summary** of the works analyzed in this study. This allows researchers to verify the metadata and scope of the research without violating copyright laws.

---

## 3. Setup & Installation

Follow these steps to configure your system and Python environment.



### (1) System-Level Dependencies (Linux/Ubuntu)
The MeCab engine must be installed at the OS level first.

```bash
# Install MeCab engine and Korean dictionaries
sudo apt-get update
sudo apt-get install -y libmecab-dev mecab-ko mecab-ko-dic
```

### (2) Environment Setup (Conda)
You can set up the environment using the provided `environment.yaml` or via manual installation.

#### Option A: Using `environment.yaml` (Recommended)
```bash
conda env create -f environment.yaml
conda activate narrative_analysis
```

#### Option B: Manual Installation
```bash
conda create -n narrative_analysis python=3.10 -y
conda activate narrative_analysis

# Build tools must be installed before other packages
pip install setuptools wheel Cython
pip install -r requirements.txt
```

---

## 4. Research Workflow & Execution



To replicate the analysis, execute the scripts in the following order:

1.  **Surprisal Extraction**:
    `python calculate_surprisal.py`
    *Calculates NLL (Negative Log-Likelihood) per sentence using Solar-10.7B.*

2.  **Semantic Metrics Extraction**:
    `python coherence_topic_calc.py`
    *Measures local coherence and global semantic shift using SBERT.*

3.  **Statistical Analysis & Visualization**:
    Open and run `MiniFiction_Analysis.ipynb`.
    *Performs clustering, sensitivity analysis, and generates research figures.*

---

## 5. Analytical Features (`MiniFiction_Analysis.ipynb`)

The final analysis stage transforms numerical vectors into narrative insights:
* **Clustering**: Groups stories into latent narrative archetypes based on trajectories.
* **Sensitivity Analysis**: Evaluates cluster stability across parameter variations (Two-way).
* **Significance Testing**: Conducts ANOVA/Kruskal-Wallis tests across categories.
* **Visual Output**: Generates heatmaps and plots in the `{project_root}/figures` directory.

---

## 6. Troubleshooting

* **Missing Data Error**: Ensure `short_novel_merged.csv` is placed in the project root before execution.
* **KSS/MeCab Errors**: If KSS fails to install, ensure `Cython` was installed first. If MeCab isn't found, verify the `apt-get` installation in Section 3-(1).
* **CUDA Mismatch**: Ensure your GPU drivers support CUDA 12.x (recommended for Ubuntu 24.04).
