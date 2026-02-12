# Model-side: Linguistic Regions and Ablation

This module includes the core model-side pipeline:

1. preprocess training data
2. accumulate gradient × parameter importance
3. extract core/language-specific regions
4. generate perturbed models
5. visualize and compare embedding behavior

## Included Core Code

- `data_preprocess/`
- `region_selection/extract_accumulated_core_linguistic_region.py`
- `region_selection/extract_language_specific_region.py`
- `training/step1_supervised_finetuning/accumulate_grad_mul_params_on_multi_languages.py`
- `training/step1_supervised_finetuning/accumulate_grad_mul_params_on_multi_languages_mistral_8b.py`
- `training/step1_supervised_finetuning/accumulate_grad_mul_params_on_multi_languages_qwen.py`
- `training/step1_supervised_finetuning/save_perturbed_model.py`
- `training/step1_supervised_finetuning/save_perturbed_model_mistral.py`
- `training/step1_supervised_finetuning/save_perturbed_model_qwen.py`
- `visualization/plot_linguistic_heatmap.py`
- `embedding_distribution/preprocess_embeddings.py`
- `embedding_distribution/reduce_dimensions.py`
- `embedding_distribution/visualize_intact_vs_ablation.py`
- `embedding_distribution/visualize_intact_vs_language_specific.py`

## Attribution

Core method is adapted from:

- [Unveiling-Linguistic-Regions-in-LLMs](https://github.com/zzhang0179/Unveiling-Linguistic-Regions-in-LLMs)

Additional contributions in this repository:

- Qwen/Mistral adaptation
- language-specific ablation design and analysis

## Minimal Workflow (Example)

1. Preprocess:
   - use scripts in `data_preprocess/`
2. Accumulate gradient × parameter:
   - run one of `accumulate_grad_mul_params_on_multi_languages*.py`
3. Select regions:
   - run `region_selection/extract_accumulated_core_linguistic_region.py`
   - run `region_selection/extract_language_specific_region.py`
4. Save perturbed model:
   - run `save_perturbed_model*.py`
5. Visualize parameter regions:
   - run `visualization/plot_linguistic_heatmap.py`
6. Analyze embedding distributions:
   - run scripts under `embedding_distribution/`

## Important

- Many training scripts are HPC-oriented and need local paths updated.
- DeepSpeed configuration is required for large-model workflows.
