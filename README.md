# Shared and Unique Neural Architecture for Language: Insights from Large Language Model Ablation

This repository is reorganized for public release and contains two major parts:

1. **Model-side linguistic region extraction and ablation**
   - Finetune or gradient accumulation to obtain parameter importance
   - Extract core linguistic regions and language-specific regions
   - Build perturbed/ablated models
   - Analyze embedding-space effects

2. **Neural encoding and visualization**
   - Extract language model features
   - Run encoding experiments on brain data
   - Compare intact vs. ablated models with cross-language/cross-model analysis

## Repository Structure

- `model_side/`
  - `data_preprocess/`: preprocessing scripts
  - `training/`: gradient accumulation and perturbation scripts
  - `region_selection/`: core/language-specific region extraction
  - `visualization/`: parameter-level heatmap visualization
  - `embedding_distribution/`: embedding distribution and ablation visualization
- `neural_encoding/`
  - `code/`: encoding and analysis scripts
  - `mats/aligned-transcripts/`: aligned transcript resources for feature extraction
- `probing/SentEval/`
  - probing evaluation framework and scripts
- `docs/`
  - optional extra notes
- `THIRD_PARTY_NOTICES.md`

## Attribution

This project is built on and adapted from:

- Unveiling Linguistic Regions in LLMs (ACL 2024): [zzhang0179/Unveiling-Linguistic-Regions-in-LLMs](https://github.com/zzhang0179/Unveiling-Linguistic-Regions-in-LLMs)
- Cross-lingual neural encoding framework: [zaidzada/crosslingual-convergence](https://github.com/zaidzada/crosslingual-convergence)
- Probing framework: [facebookresearch/SentEval](https://github.com/facebookresearch/SentEval)

We additionally adapted pipelines for **Qwen** and **Mistral**, and implemented an additional **language-specific ablation** workflow.

## Quick Start

### 1) Model-side

See `model_side/README.md`.

### 2) Neural encoding

See `neural_encoding/README.md`.

### 3) Probing

See `probing/SentEval/README.md`.

## Notes

- Some scripts still require local dataset/model paths (HPC paths) to be provided by arguments or environment variables.
- For `encoding_MNI152.py`, set `MNI152_MASK_PATH` if your mask is not under `neural_encoding/resources/`.
