"""Encoding with MNI152 Gray Matter Mask"""

import warnings
from collections import defaultdict
import os
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
import torch
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from nilearn.maskers import NiftiMasker
from scipy.stats import zscore
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util.constants import (
    LANGS,
    N_RUNS,
    RAW_BOLD_SLICES,
    RUNS,
    SUB_TO_RUN,
    SUBS,
    TR,
    TRIAL_TRS,
)
from util.path import Path
from util.subject import get_bold_path, load_bold
from voxelwise_tutorials.delayer import Delayer

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
sklearn.set_config(assume_finite=True)


def resolve_mni152_mask_path() -> str:
    """Resolve the MNI152 gray-matter mask path with environment override support."""
    env_path = os.environ.get("MNI152_MASK_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "neural_encoding" / "resources" / "MNI152_template_gm_mask_2mm.nii.gz",
        repo_root / "brain_encoding" / "atlas" / "MNI152_template_gm_mask_2mm.nii.gz",
        Path("brain_encoding/atlas/MNI152_template_gm_mask_2mm.nii.gz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "MNI152 gray-matter mask not found. Set MNI152_MASK_PATH or place "
        "MNI152_template_gm_mask_2mm.nii.gz under neural_encoding/resources/."
    )


def get_sub_bold(sub: str, lang: str, space: str, **kwargs) -> np.ndarray:

    bold_path = get_bold_path(space=space, **kwargs)
    bold_path.update(sub=sub, task=f"lpp{lang}")

    # temporary fix
    if space.startswith("fs"):
        bold_path.update(space="schaefer1k")

    bold = []
    for run in range(N_RUNS):
        sub_run = SUB_TO_RUN[sub][run]
        bold_path.update(run=sub_run)
        data = load_bold(bold_path)
        if space.startswith("fs"):  # NOTE
            data = data[RAW_BOLD_SLICES[lang]]
        elif space.startswith("MNIColin"):
            data = zscore(data, axis=-1)
        bold.append(data)
    if space.startswith("MNI"):
        bold = np.concatenate(bold, axis=-1)
    else:
        bold = np.vstack(bold)
    return bold


def get_whisper_features(lang: str, modelname: str, **kwargs):
    audio_path = Path(root=f"mats/{modelname}", task=f"lpp{lang}", run="X", ext="npy")
    features = []
    for run in RUNS:
        audio_path.update(run=run)
        feature = np.load(audio_path)

        n_trs = TRIAL_TRS[lang][run - 1]
        n_features = len(feature)

        if n_features > n_trs:
            tr_feature = feature[:n_trs]
        elif n_features < n_trs:
            tr_feature = np.zeros((n_trs, feature.shape[-1]), dtype=feature.dtype)
            tr_feature[:n_features] = feature
        else:
            tr_feature = feature

        features.append(tr_feature)

    features = np.vstack(features)
    return None, features


def get_transcript_features(
    lang: str, modelname: str, layer: int = -1, downsample: bool = True, **kwargs
):
    # reduces transcript over tokens
    emb_path = Path(root=f"scratch_link/embeddings/{modelname}", task=f"lpp{lang}", run="X", ext="npy")
    transcript_path = Path(root=f"scratch_link/embeddings/{modelname}", task=f"lpp{lang}", ext="csv")

    cols = ["word_idx", "word", "onset", "offset", "section", "sentence_index"]
    df_transcript = pd.read_csv(transcript_path, usecols=cols)

    features = []
    transcripts = []
    for run in RUNS:
        emb_path.update(run=run)
        feature = np.load(emb_path.fpath, mmap_mode="r")
        # The new embeddings are from the last layer and are not layered.
        # So we remove the layer indexing.
        # feature = feature[layer]
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)


        df_run = df_transcript[df_transcript.section == run]
        df_run.reset_index(drop=True, inplace=True)
        assert len(df_run) == len(
            feature
        ), f"Run {run}: Mismatch between transcript rows ({len(df_run)}) and embeddings ({len(feature)})"

        reduce_features = []
        for _, df_word in df_run.groupby("word_idx"):
            reduce_features.append(feature[df_word.index].mean(0))
        features.append(np.stack(reduce_features))
        transcripts.append(df_run.groupby("word_idx").first())

    transcript = pd.concat(transcripts)
    features = np.vstack(features)

    if downsample:
        features = downsample_features(lang, transcript, features)

    return transcript, features


def downsample_features(language, transcript_df, features):
    feature_dim = features.shape[-1]
    transcript_df["TR"] = transcript_df.onset.divide(TR).apply(np.floor).apply(int)

    new_features = []
    for section, section_df in transcript_df.groupby("section"):
        n_trs = TRIAL_TRS[language][section - 1]
        run_features = np.zeros((n_trs, feature_dim), dtype=features.dtype)
        for tr in range(n_trs):
            tr_mask = section_df["TR"] == tr
            if tr_mask.any():
                tr_indices = section_df.index[tr_mask]
                run_features[tr] = features[tr_indices, ...].mean(0)
        new_features.append(run_features)

    return np.vstack(new_features)


def get_features(lang: str, modelname: str, **kwargs):
    if "spectral" in modelname or "whisper" in modelname:
        return get_whisper_features(lang, modelname, **kwargs)
    else:
        return get_transcript_features(lang, modelname, **kwargs)


def main(
    language: list[str],
    modelname: str,
    space: str,
    layer: int,
    alphas: list[float],
    suffix: str,
    device: str,
    save_model: bool,
    **kwargs,
):

    if device == "cuda":
        set_backend("torch_cuda")

    # Load the MNI152 gray matter mask
    mni152_mask_path = resolve_mni152_mask_path()
    mni152_mask_img = nib.load(mni152_mask_path)
    mni152_mask_data = mni152_mask_img.get_fdata().astype(np.float32)
    
    # The MNI152 gray matter mask is already binary (0 and 1), so we can use it directly
    # Convert to int8 for consistency with the original code
    binary_mask_data = mni152_mask_data.astype(np.int8)
    binary_mask_img = nib.Nifti1Image(
        binary_mask_data, mni152_mask_img.affine, mni152_mask_img.header
    )

    # Use the MNI152 gray matter mask with NiftiMasker
    masker = NiftiMasker(mask_img=binary_mask_img).fit()
    affine = mni152_mask_img.affine

    for lang in tqdm(language, desc="lang", ncols=80):

        _, features = get_features(lang, modelname)

        lang_runs = TRIAL_TRS[lang]
        run_ids = np.repeat(np.arange(N_RUNS), lang_runs)

        print(f"lang: {lang}, modelname: {modelname}")

        for sub in tqdm(SUBS[lang], desc="sub ", leave=False, ncols=80):

            output_path = Path(
                root=f"scratch_link/result_MNI152/{lang}/{modelname}{suffix}",
                sub=sub,
                desc=sub,
                ext="npz",
            )
            output_path.mkdirs()

            X = features
            Y = get_sub_bold(sub, lang, space, **kwargs)
            # for volumetric data
            Y = masker.transform(nib.Nifti1Image(Y, affine))
            Y = Y.astype(np.float32)

            # Prepare model
            delays = [1, 2, 3, 4]
            n_samples = len(X)
            n_features = X.shape[1] * len(delays)
            inner_cv = KFold(n_splits=2, shuffle=False)
            solver_params = {"n_targets_batch": 10000, "diagonalize_method": "svd"}
            if n_samples > n_features:
                solver = RidgeCV(alphas, fit_intercept=True, cv=inner_cv, solver_params=solver_params)
            else:
                # solver = KernelRidgeCV(alphas, fit_intercept=True, cv=inner_cv)
                # cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED
                # 使用SVD方法避免CUDA特征值分解失败
                solver = KernelRidgeCV(
                    alphas, 
                    fit_intercept=True, 
                    cv=inner_cv,
                    solver_params=solver_params
                )
            pipeline = make_pipeline(StandardScaler(), Delayer(delays=delays), solver)

            # train model
            result = {}
            results = defaultdict(list)
            cv = LeaveOneGroupOut()
            for k, (train_index, test_index) in enumerate(cv.split(X, Y, run_ids)):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                pipeline.fit(X_train, Y_train)

                Y_preds = pipeline.predict(X_test)
                scores = correlation_score(Y_test, Y_preds)
                results["cv_scores"].append(scores.numpy(force=True))
                result[f"preds_run-{k}"] = Y_preds.numpy(force=True)

                if save_model:
                    model_path = output_path.copy()
                    model_path.update(sub=sub, desc=str(k), suffix="model", ext=".pkl")
                    joblib.dump(pipeline, model_path.fpath)

            result["cv_scores"] = np.stack(results["cv_scores"])
            np.savez(output_path, **result)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("-p", "--space", default="MNIColin27")
    parser.add_argument("-m", "--modelname", required=True)
    parser.add_argument("--layer", type=int)  # , required=True) only if decoder
    parser.add_argument("--desc", type=str)
    parser.add_argument("--hemi", type=str)
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("-l", "--language", nargs="+", choices=LANGS, default=LANGS)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--alphas", default=np.logspace(0, 19, 20))
    _args = parser.parse_args()

    if torch.cuda.is_available() and not _args.force_cpu and _args.device == "cpu":
        _args.device = "cuda"
    main(**vars(_args)) 