import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from util.constants import LANGS
from util.path import Path
from util.transcript import get_aligned_transcript

torch.set_grad_enabled(False)


def main(
    language: list[str],
    modelname: str,
    model_alias: str,
    tokenizer_name_or_path: str | None,
    untrained: bool,
    device: str,
    **kwargs,
):

    if model_alias is None:
        model_alias = modelname.split("/")[-1]

    output_path = Path(root=f"scratch_link/embeddings/{model_alias}", task="lppXX", run="X", ext="npy")

    tokenizer_source = tokenizer_name_or_path or modelname
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    
    # LLaMA2 specific: Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if untrained:
        set_seed(42)
        config = AutoConfig.from_pretrained(modelname, output_hidden_states=True)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(modelname, output_hidden_states=True)

    print(
        f"Model : {modelname}"
        f"\nModel class : {type(model)}"
        f"\nLayers: {model.config.num_hidden_layers}"
        f"\nEmbDim: {model.config.hidden_size}"
    )
    model = model.eval()
    model = model.to(device)

    for lang in tqdm(language, desc="lang", ncols=80):
        # LLaMA2 uses BOS token as the beginning of sequence
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        transcript_df = get_aligned_transcript(lang)
        transcript_df.insert(0, "word_idx", transcript_df.index.values)
        transcript_df["hftoken"] = transcript_df.word.apply(
            lambda x: tokenizer.tokenize(str(x))
        )
        transcript_df = transcript_df.explode("hftoken", ignore_index=True)
        transcript_df["token_id"] = transcript_df.hftoken.apply(
            tokenizer.convert_tokens_to_ids
        )

        for run, section_df in tqdm(
            transcript_df.groupby("section"), desc="run ", leave=False, ncols=80
        ):
            states = []
            for _, sentence in section_df.groupby("sentence_index"):
                n_sentence_tokens = len(sentence)
                # LLaMA2 has max context of 4096, but we'll use similar window as BERT for consistency
                # Account for BOS and EOS tokens (2 special tokens instead of 3)
                n_context_tokens = 510 - n_sentence_tokens  # 512 - 2 special tokens
                sentence_start_id = sentence.index[0].item()
                context_start_id = max(0, sentence_start_id - n_context_tokens)
                context = section_df.iloc[context_start_id:sentence_start_id]

                # LLaMA2 input format: 
                example = torch.tensor(
                    [
                        [bos_token_id]
                        + context["token_id"].to_list()
                        + sentence["token_id"].to_list()
                        + [eos_token_id]
                    ]
                )

                outputs = model(input_ids=example.to(device))
                
                # Extract hidden states for the sentence tokens
                # In LLaMA2, we need to account for the different structure
                # The sentence starts after BOS + context tokens
                sentence_start_pos = 1 + len(context)  # 1 for BOS token
                sentence_end_pos = sentence_start_pos + len(sentence)
                
                last_hidden_state = outputs["hidden_states"][-1]
                sentence_states = last_hidden_state[:, sentence_start_pos:sentence_end_pos, :]
                states.append(sentence_states.numpy(force=True))

            embeddings = np.concatenate(states, axis=1)

            output_path.update(task=f"lpp{lang}", run=f"{run}")
            output_path.mkdirs()
            np.save(output_path, embeddings)

        transcript_path = output_path.copy().update(ext="csv")
        del transcript_path["run"]
        transcript_df.to_csv(transcript_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", required=True)
    parser.add_argument("-a", "--model-alias", required=True)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("-l", "--language", nargs="+", choices=LANGS, default=LANGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    _args = parser.parse_args()

    if torch.cuda.is_available() and not _args.force_cpu and _args.device == "cpu":
        _args.device = "cuda"
    main(**vars(_args))