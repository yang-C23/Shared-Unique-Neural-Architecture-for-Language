# Neural Encoding

This module contains the encoding framework and visualization scripts for comparing intact vs. ablated models in neural encoding tasks.

## Included Core Code

- `code/encoding_MNI152.py`
- `code/features_llama.py`
- `code/features_mistral.py`
- `code/features_qwen.py`
- `code/plot_cross_language_rmap_average.py`
- `code/plot_cross_language_tmap_analysis.py`
- `code/plot_average_core-language-ablation_cross_6model_tmaps.py`
- `code/extract_encoding_results.py`
- `code/visualize_encoding_results.py`
- `code/plot_lpi_analysis.py`
- `code/plot_cross_model_lpi_average.py`
- `code/util/`
- `mats/aligned-transcripts/`

## Attribution

Framework adapted from:

- [crosslingual-convergence](https://github.com/zaidzada/crosslingual-convergence)

## Path/Runtime Notes

- `code/util/transcript.py` is adapted to this repository layout.
- `encoding_MNI152.py` resolves mask path in this order:
  1. `MNI152_MASK_PATH` env var
  2. `neural_encoding/resources/MNI152_template_gm_mask_2mm.nii.gz`
  3. `brain_encoding/atlas/MNI152_template_gm_mask_2mm.nii.gz`
- `features_{llama,mistral,qwen}.py` now support `--tokenizer-name-or-path`.

## Example

```bash
python neural_encoding/code/features_qwen.py \
  -m /path/to/model \
  -a qwen_model_alias \
  --tokenizer-name-or-path /path/to/tokenizer \
  -l EN CN FR
```
