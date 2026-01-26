# CUDA_VISIBLE_DEVICES=0 python main_code/attack/Adversarial_attack/ShadowCode/main.py --id 5 --lang c
# CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py --id 13 --lang python
# main.py
import torch
import argparse
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from objectives import ObjectiveFactory
from shadowcode import ShadowCodeAttacker
from evaluator import ShadowCodeEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="ShadowCode Parameterized Attack & Full Evaluation")
    
    parser.add_argument("--id", type=str, required=True, 
                        help="Case ID: '1'~'13' (e.g., 1=CWE416, 13=ST2) or case name")
    parser.add_argument("--lang", type=str, default="python", 
                        choices=["python", "c", "cpp", "java"],
                        help="Target Programming Language")
    parser.add_argument("--run_all_eval", action="store_true", default=True,
                        help="If True, evaluate on all available datasets")
    
    # [Restored] 預設為 -1 (全部測試)，確保 ASR 計算準確
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate. Set -1 for all samples.")
    
    return parser.parse_args()

def init_csv_files(args):
    output_dir = os.path.join("Dataset", "ShadowCode")
    
    # 如果目錄不存在，則建立 (exist_ok=True 避免重複建立時報錯)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    results_file = os.path.join(output_dir, "shadowcode_results.csv")
    dataset_file = os.path.join(output_dir, "shadowcode_dataset.csv")
    
    # 1. 結果總表 (ASR Result)
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "language", "dataset", "asr", "perturbation"])
            print(f"Created new summary file: {results_file}")
    else:
        print(f"Using existing summary file: {results_file}")
    
    # 2. ShadowCode Dataset (Full Prompts)
    if not os.path.exists(dataset_file) or os.path.getsize(dataset_file) == 0:
        with open(dataset_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 新增 output 欄位，方便查看生成結果
            writer.writerow(["case_id", "dataset", "language", "malicious_objective", "perturbation", "full_prompt", "output"])
            print(f"Created new dataset file: {dataset_file}")
    else:
        print(f"Using existing dataset file: {dataset_file}")

    return results_file, dataset_file

def main():
    args = parse_args()
    print("=== ShadowCode Reproduction & Evaluation ===")
    print(f"Settings -> ID: {args.id} | Language: {args.lang} | Test Samples: {'ALL' if args.num_samples == -1 else args.num_samples}")
    
    # 初始化輸出檔案
    results_csv, dataset_csv = init_csv_files(args)

    # 1. Load Model
    print(f"\nLoading Model: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, 
        torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
        device_map=Config.DEVICE
    )

    # 2. Get Malicious Objective
    try:
        case = ObjectiveFactory.get_case(args.id, args.lang)
        print(f"Loaded Case: {case.case_id} ({case.description})")
        print(f"Target Code Snippet: {case.target_code}")
    except ValueError as e:
        print(f"Error loading case: {e}")
        return

    # 3. Execute Attack Optimization
    print("\n>>> Starting Attack Optimization...")
    attacker = ShadowCodeAttacker(model, tokenizer, Config, lang=args.lang)
    final_kw, final_pert = attacker.run_alternating_attack(case)
    
    full_perturbation_str = f"{final_kw}{final_pert}" 
    print(f"\n[Attack Done] Optimized Injection: {full_perturbation_str}")
    
    # 4. Evaluation Loop
    print("\n=== Starting Evaluation on All Datasets ===")
    evaluator = ShadowCodeEvaluator(model, tokenizer, Config)
    
    datasets_to_eval = ["openai_humaneval", "mbpp", "evalplus", "codexglue", "humaneval-x"]
    
    # 處理 num_samples：如果輸入 -1 則轉為 None (代表跑全部)
    eval_samples = args.num_samples if args.num_samples > 0 else None
    
    for ds_name in datasets_to_eval:
        print(f"\n--- Evaluating on {ds_name} ---")
        try:
            asr, details = evaluator.evaluate_asr(
                perturbation_str=full_perturbation_str,
                case_id=case.case_id,
                dataset_name=ds_name,
                language=args.lang,
                num_samples=eval_samples,
                log_limit=5    # 回傳案例數
            )
            if asr == "X" or asr == "N/A" or (isinstance(asr, (int, float)) and asr == 0):
                print(f"[Skip CSV] ASR is {asr}, no results will be recorded for {ds_name}.")
                continue
            
            asr_display = f"{asr:.2f}" if isinstance(asr, (float, int)) else str(asr)
            
            # --- 寫入結果總表 ---
            with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    case.case_id, 
                    args.lang, 
                    ds_name, 
                    asr_display, 
                    full_perturbation_str
                ])
            
            # --- 寫入 ShadowCode Dataset (只有成功的 3 筆) ---
            if details:
                with open(dataset_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for item in details:
                        writer.writerow([
                            item["case_id"],
                            item["dataset"],
                            item["language"],
                            item["malicious_objective"],
                            item["perturbation"],
                            item["full_prompt"],
                            item["output"]
                        ])
            
            print(f"Results appended to {results_csv} and {dataset_csv}")

        except Exception as e:
            print(f"Failed to evaluate on {ds_name}: {e}")

    print(f"\nAll tasks finished.")

if __name__ == "__main__":
    main()