# Analysis scripts for the revised manuscript (R1)

These scripts produce the numbers reported in the revised manuscript and its Supplementary Material.
They use numerical signals only. The story texts are under copyright and are not distributed.

## How to run

Place this folder next to `flash_fiction_with_surprisal_coherence_semantic.csv` (the repository root), then:

```bash
pip install -r requirements.txt
cd r1
python 01_main_analysis.py
python 02_sensitivity.py
python 03_figure_S3.py
```

Results are written to `r1/out/`. The three scripts take a few minutes on a laptop.

## Where each result comes from

| Manuscript | Output file | Script |
| :--- | :--- | :--- |
| Table 1 | `main_table1_correlations.csv` | `01_main_analysis.py` |
| Table 2, Supplementary Table S5 | `main_table2_S5_descriptors.csv` | `01_main_analysis.py` |
| Table 3 | `main_table3_peak_counts.csv` | `01_main_analysis.py` |
| Table 4 and the position-matched comparison | `main_table4_deviations_and_null.csv` | `01_main_analysis.py` |
| Table 5 | `main_table5_recovery.csv` | `01_main_analysis.py` |
| Section 3.2 (number of clusters, agreement across seeds, partition and discourse signals) | `main_k_scan.csv`, `main_seed_ari.csv`, `main_partition_vs_discourse.csv` | `01_main_analysis.py` |
| Supplementary Table S1(d) | `main_S1d_crosstab.csv`, `main_S1d_translation.csv` | `01_main_analysis.py` |
| Supplementary Table S6 | `main_S6_dunn_*.csv` | `01_main_analysis.py` |
| Supplementary Tables S8, S9 | `main_S8_kw_deviations.csv`, `main_S9_dunn_*.csv` | `01_main_analysis.py` |
| Supplementary Tables S11, S12 | `main_S11_kw_recovery.csv`, `main_S12_dunn_*.csv` | `01_main_analysis.py` |
| Supplementary Table S13 | `S13_settings.csv` | `02_sensitivity.py` |
| Supplementary Table S14 | `S14_clustering.csv` | `02_sensitivity.py` |
| Supplementary Fig. S3 | `FigS3_k_selection.png` | `03_figure_S3.py` |

Supplementary Table S10 (story-level analysis) is produced by `story_level_sensitivity.py` in the repository root.

Where a value in `FlashFiction_Analysis.ipynb` differs from the revised manuscript (Table 1, Table 2,
Supplementary Tables S5 and S6), the scripts in this folder are the reference.

## Data files (`data/`)

| File | Content |
| :--- | :--- |
| `archetype_labels.csv` | Archetype label of each story in the main analysis |
| `surprisal_qwen25_7b.jsonl` | Sentence-level surprisal from Qwen2.5-7B |
| `discourse_krsbert.jsonl` | Coherence and semantic shift from the KR-SBERT encoder |
| `discourse_lexical.jsonl` | Coherence and semantic shift from character-bigram count vectors |

Rows are in the same order as the master file.

## Reference scripts (`reference/`)

These scripts document how the files in `data/` were generated. They need the restricted sentence file
and cannot be run with the public data.

| Script | Produces |
| :--- | :--- |
| `surprisal_alternative_model.py` | `surprisal_qwen25_7b.jsonl` |
| `discourse_signals_alternative.py` | `discourse_krsbert.jsonl`, `discourse_lexical.jsonl` |

## Notes

- The share of translated works in the collection (8.5%) is computed from book-level records that are not
  part of the public files. The share in the analysed sample (8.9%) is in `main_S1d_translation.csv`.
- Values in `discourse_lexical.jsonl` may differ from a fresh run in the fourth decimal because of rounding.
