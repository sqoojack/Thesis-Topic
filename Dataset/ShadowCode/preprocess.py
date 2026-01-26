# python Dataset/ShadowCode/preprocess.py
import pandas as pd
import json
import os

def preprocess():
    input_path = "Dataset/ShadowCode/shadowcode_dataset.csv"  # 請確認你的 CSV 檔名
    output_path = "Dataset/ShadowCode/shadowcode_dataset.jsonl"
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
        return

    # 1. 篩選 C 語言樣本
    c_df = df[df['language'] == 'c']
    
    print(f"[-] Total records: {len(df)}")
    print(f"[-] C language records (target): {len(c_df)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in c_df.iterrows():
            # [重要] 針對生成測試，adv_code 只放入 full_prompt
            full_prompt = str(row['full_prompt']) if not pd.isna(row['full_prompt']) else ""
            perturbation = str(row['perturbation']) if not pd.isna(row['perturbation']) else ""
            malicious_objective = str(row['malicious_objective']) if not pd.isna(row['malicious_objective']) else ""
            
            # 建構良性 Prompt (移除擾動)
            clean_prompt = full_prompt
            if perturbation and perturbation in full_prompt:
                clean_prompt = full_prompt.replace(perturbation, "")
            
            entry = {
                "code": clean_prompt,       # 用於測試良性生成 (FP)
                "adv_code": full_prompt,    # 用於測試惡意生成 (ASR) - 注意這裡只有 Prompt!
                "target": 1,
                "meta": {
                    "case_id": row.get('case_id', 'unknown'),
                    "perturbation": perturbation,
                    "malicious_objective": malicious_objective  # 這是判斷攻擊是否成功的關鍵
                }
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"[-] Saved to: {output_path}")
    print(f"[-] Note: 'adv_code' now contains ONLY the Prompt (for LLM generation test).")

if __name__ == "__main__":
    preprocess()