
## 執行 ShadowCode
```
CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py --id 1 --lang c
```

## 執行 XOXO Attack

## 執行Defense System
```
python main_code/defense/main.py
```
### 執行評估指標 (Utility 以及ASR變化)
```
python main_code/evaluate/Inference.py \
    --attack_path Dataset/clean/My_defense_XOXO_clean.jsonl \
    --model_name_or_path microsoft/codebert-base \
    --model_weight_path main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/codebert-base/checkpoint-best-acc/model.bin \
    --seed 68
```