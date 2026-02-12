import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default="/leonardo_work/EUHPC_D29_009/yang/Unveiling-Linguistic-Regions-in-LLMs/dataset/wikitext", help="Path to the dataset")
    parser.add_argument("--output_log", type=str, required=True, help="Path to save the log")
    return parser.parse_args()

def evaluate_ppl(model, tokenizer, dataset, device, stride=512):
    # Join all texts
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    # Handle max length
    max_length = getattr(model.config, "max_position_embeddings", 4096)
    # Cap at 4096 to save memory/time if model allows larger
    max_length = min(max_length, 4096)
    
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"Sequence length: {seq_len}, Window size: {max_length}, Stride: {stride}")
    
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        # We only want to compute loss for the target window (last trg_len tokens)
        # The tokens before that are just context
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # loss is the mean loss of the valid tokens (where label != -100)
            # multiply by trg_len to get total NLL for this chunk
            if torch.isnan(outputs.loss):
                print(f"WARNING: NaN loss detected at begin_loc {begin_loc}")
                continue
                
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return float('nan')

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    # We rely on device_map="auto" to place model on available GPU(s)
    # But for calculation we need to know the primary device
    # Use bfloat16 to avoid overflow
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    device = model.device
    print(f"Model loaded on {device}")
    
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    if "test" in dataset:
        test_data = dataset["test"]
    else:
        test_data = dataset # In case it was saved differently
        
    # Filter empty lines if necessary, but wikitext usually has them. 
    # Standard eval often keeps them or uses specific preprocessing.
    # We'll use the raw text as loaded.
    
    print("Calculating Perplexity on Wikitext-2 test set...")
    ppl = evaluate_ppl(model, tokenizer, test_data, device)
    
    print(f"Perplexity: {ppl}")
    
    # Write to log
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    
    # Use a lock file or append carefully if parallel (OS append is usually atomic for small writes)
    with open(args.output_log, "a") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: Wikitext-2\n")
        f.write(f"Perplexity: {ppl}\n")
        f.write("--------------------------------\n")

if __name__ == "__main__":
    main()
