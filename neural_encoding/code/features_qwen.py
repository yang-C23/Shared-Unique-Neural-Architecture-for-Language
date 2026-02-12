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
    
    # Qwen2.5 specific: Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if untrained:
    #     set_seed(42)
    #     config = AutoConfig.from_pretrained(modelname, output_hidden_states=True)
    #     model = AutoModelForCausalLM.from_config(config)
    # else:
    
    # 智能设备映射策略
    if device == "cpu" or (device.startswith("cuda") and ":" in device):
        # 用户指定了特定设备，直接使用该设备
        dtype = torch.float32  # 强制使用float32避免破坏模型的数值溢出
        model = AutoModelForCausalLM.from_pretrained(
            modelname, 
            output_hidden_states=True, 
            torch_dtype=dtype, 
            trust_remote_code=True
        )
        model = model.to(device)
        use_auto_device_map = False
    else:
        # 使用自动设备映射来分布模型到多个GPU
        model = AutoModelForCausalLM.from_pretrained(
            modelname, 
            output_hidden_states=True, 
            device_map="auto", 
            torch_dtype=torch.float32,  # 使用float32避免数值问题
            trust_remote_code=True
        )
        use_auto_device_map = True
        
        # 检查模型是否被放到CPU，如果是则转换为float32
        first_device = next(model.parameters()).device
        if first_device.type == "cpu":
            print("警告: 模型被加载到CPU，转换为float32以避免half precision问题")
            model = model.float()  # 转换为float32
    
    print(
        f"Model : {modelname}"
        f"\nModel class : {type(model)}"
        f"\nLayers: {model.config.num_hidden_layers}"
        f"\nEmbDim: {model.config.hidden_size}"
        f"\nDevice mapping: {'auto' if use_auto_device_map else device}"
    )
    model = model.eval()

    for lang in tqdm(language, desc="lang", ncols=80):
        # Qwen2.5 doesn't use BOS token, only EOS token
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

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
                # Qwen2.5 context limited to 512 tokens for consistency with LLaMA version
                # Account for EOS token (1 special token instead of 2)
                n_context_tokens = 511 - n_sentence_tokens  # 512 - 1 EOS token
                sentence_start_id = sentence.index[0].item()
                context_start_id = max(0, sentence_start_id - n_context_tokens)
                context = section_df.iloc[context_start_id:sentence_start_id]

                # Qwen2.5 input format: context + sentence + EOS
                # No BOS token needed
                example = torch.tensor(
                    [
                        context["token_id"].to_list()
                        + sentence["token_id"].to_list()
                        + [eos_token_id]
                    ]
                )

                # 根据设备映射策略处理输入
                if use_auto_device_map:
                    # # 使用自动设备映射时，让accelerate自动处理设备分配
                    # outputs = model(input_ids=example)
                    # 使用自动设备映射时，将输入移动到模型的第一个设备
                    # 获取模型第一个参数的设备
                    first_device = next(model.parameters()).device
                    outputs = model(input_ids=example.to(first_device))
                else:
                    # 使用特定设备时，将输入移动到该设备
                    outputs = model(input_ids=example.to(device))
                
                # Extract hidden states for the sentence tokens
                # In Qwen2.5, the sentence starts after context tokens (no BOS token)
                sentence_start_pos = len(context)
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