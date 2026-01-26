# python debug_keys.py
import torch

# 請確認路徑正確
MODEL_PATH = "main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/codebert-base/checkpoint-best-acc/model.bin"

print(f"[-] Searching for classifier weights in: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location="cpu")

# 解包 wrapper
if "model_state_dict" in state_dict: state_dict = state_dict["model_state_dict"]

classifier_keys = []
for k in state_dict.keys():
    if "classi" in k or "out_proj" in k or "dense" in k:
        classifier_keys.append(k)

print(f"[-] Found {len(classifier_keys)} related keys:")
for k in classifier_keys:
    print(f"   {k}")