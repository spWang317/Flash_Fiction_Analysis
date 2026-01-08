# Korean Flash Fiction Narrative Analysis

This project provides a computational framework to analyze the narrative structures of Korean short stories (flash fiction). It utilizes Large Language Models (LLMs) to calculate **Surprisal**, and Sentence-BERT (SBERT) to measure **Local Coherence** and **Global Semantic Shift**.

---

## 1. Setup & Installation Guide

This guide ensures consistent reproducibility across different research environments. Follow the steps below to configure your system.

### (1) System-Level Dependencies (Linux/Ubuntu)

This project requires the **MeCab** morphological analyzer engine. Since MeCab is a system-level tool written in C++, it must be installed via the package manager before setting up the Python environment.

```bash
# Update package list and install MeCab system libraries
sudo apt-get update
sudo apt-get install -y libmecab-dev mecab-ko mecab-ko-dic
```

---

### (2) Data Source & Policy

#### `short_novel_merged.csv` (Not Provided)
This is the primary input file containing the narrative data. Due to **copyright restrictions** regarding the proprietary literary content, the full text dataset is **not publicly provided** in this repository.

The schema for this file is as follows:

| Column Name | Description |
|:---|:---|
| `isbn` | International Standard Book Number |
| `story_title` | Title of the individual flash fiction story |
| `text` | **Full narrative text** (Used for surprisal/coherence calculation) |
| `author` | Name of the author |
| `birth_year` | Birth year of the author |
| `gender` | Gender of the author |
| `book_title` | Title of the source book or anthology |
| `publisher` | Name of the publisher |
| `pub_year` | Year of publication |
| `country` | Country of author |

#### `book_list_summary.csv` (Provided)
In place of the full text data, we provide a **bibliography summary** of the works included in this research. This file allows researchers to verify the metadata of the stories analyzed without violating copyright.



---

### (3) Environment Setup (Conda)

You can set up the environment using either `environment.yaml` (recommended) or manual installation.

#### Option A: Using `environment.yaml`
```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate narrative_analysis
```

#### Option B: Manual Setup
```bash
# Create and activate environment
conda create -n narrative_analysis python=3.10 -y
conda activate narrative_analysis

# Install build tools first (Critical for KSS/MeCab-python3)
pip install setuptools wheel Cython

# Install dependencies
pip install -r requirements.txt
```

---

### (4) Execution Order & Workflow



To replicate the study, follow the pipeline in this specific order:

1.  **Feature Extraction - Surprisal**:
    `python calculate_surprisal.py`
    *Outputs Negative Log-Likelihood (NLL) vectors using Solar-10.7B.*

2.  **Feature Extraction - Coherence & Semantic Shift**:
    `python coherence_topic_calc.py`
    *Measures adjacent similarity and global context deviation using SBERT.*

3.  **Narrative Archetype Analysis**:
    Open and run `MiniFiction_Analysis.ipynb`.
    *Performs clustering, sensitivity analysis, and statistical validation (ANOVA/Kruskal-Wallis).*

---

### (5) Analytical Features (`MiniFiction_Analysis.ipynb`)

The analysis notebook transforms numerical vectors into research insights:

* **Clustering**: Groups stories into latent narrative archetypes.
* **Two-way Sensitivity Analysis**: Evaluates the stability of clusters against parameter variations.
* **Statistical Validation**: Conducts significance testing across narrative categories.
* **Visualization**: Generates heatmaps and plots saved in the `{project_root}/figures` directory.

---

### (6) Troubleshooting

* **Copyright Issues**: If you wish to test the code, ensure you prepare a CSV file (`short_novel_merged.csv`) matching the schema described in section (2).
* **MeCab Errors**: Ensure you ran the `sudo apt-get` commands. MeCab must be installed at the OS level.
* **KSS Installation**: If `pip install` fails for KSS, verify that `Cython` was installed successfully beforehand.
