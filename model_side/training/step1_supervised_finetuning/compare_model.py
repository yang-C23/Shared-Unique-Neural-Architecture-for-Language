import torch
from transformers import AutoModelForCausalLM

def compare_models(model_dir_1, model_dir_2):
    # 加载两个模型
    model1 = AutoModelForCausalLM.from_pretrained(model_dir_1, torch_dtype=torch.float32)
    model2 = AutoModelForCausalLM.from_pretrained(model_dir_2, torch_dtype=torch.float32)
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 首先判断参数名是否完全一致
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    if keys1 != keys2:
        print("模型参数名不一致！")
        print("model1独有参数：", keys1 - keys2)
        print("model2独有参数：", keys2 - keys1)
        return False

    all_equal = True
    for key in state_dict1.keys():
        t1 = state_dict1[key]
        t2 = state_dict2[key]
        if not torch.equal(t1, t2):
            print(f"参数 {key} 不一致")
            all_equal = False
        # 若想查看具体不一致数目
        # print(f"{key} 相等元素数: {(t1==t2).sum().item()}/{t1.numel()}")

    if all_equal:
        print("两个模型权重完全一致！")
    else:
        print("有参数不一致！")
    return all_equal

# 使用示例
compare_models('/mnt/iusers01/fatpou01/compsci01/v07051yc/Unveiling-Linguistic-Regions-in-LLMs/ragion_freeze_path_to_save/llama_break_top001_perturbed_saved', '/mnt/iusers01/fatpou01/compsci01/v07051yc/Unveiling-Linguistic-Regions-in-LLMs/ragion_freeze_path_to_save/llama_break_top001')
