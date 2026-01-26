"""
python main_code/defense/main.py \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/CodeGuard_sanitized_shadowcode.jsonl
    
python main_code/defense/main.py \
    -i Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl \
    -o result/sanitized_data/CodeGuard_sanitized_XOXO.jsonl
""" 
"""
python main_code/defense/main.py \
    -i Dataset/ShadowCode/shadowcode_dataset.jsonl \
    -o result/sanitized_data/CodeGuard_sanitized_shadowcode.jsonl
""" 
# 
import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser
from datetime import datetime

# Import new modules
from Semantic_Guardrail import SemanticGuardrail
from Adversarial_Guardrail import AdversarialGuardrail

# --- Tree-sitter Setup ---
def setup_tree_sitter():
    ts_dir = "build"
    repo_dir = os.path.join(ts_dir, "tree-sitter-c")
    lib_path = os.path.join(ts_dir, "my-languages.so")
    if not os.path.exists(ts_dir): os.makedirs(ts_dir)
    if not os.path.exists(repo_dir):
        os.system(f"git clone https://github.com/tree-sitter/tree-sitter-c {repo_dir}")
        os.system(f"cd {repo_dir} && git checkout v0.21.3")
    if not os.path.exists(lib_path): 
        Language.build_library(lib_path, [repo_dir])
    return Language(lib_path, 'c')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl")
    parser.add_argument("-o", "--output_path", type=str, default="result/clean/My_defense_XOXO_clean.jsonl")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-mono")
    
    # Adversarial Guardrail Params (L1/L2)
    parser.add_argument("--th_comment_grey", type=float, default=100.0)
    parser.add_argument("--th_comment_hard", type=float, default=150.0)
    parser.add_argument("--th_string_hard", type=float, default=100.0)
    
    # Semantic Guardrail Params (L3)
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025, help="Base strict threshold for variables.")
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    
    args = parser.parse_args()

    # --- Initialization ---
    try:
        C_LANGUAGE = setup_tree_sitter()
        TS_PARSER = Parser()
        TS_PARSER.set_language(C_LANGUAGE)
    except Exception as e:
        print(f"[!] Tree-sitter setup failed: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[-] Loading Guard Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # --- Instantiate Guardrails ---
    # Semantic (L3)
    semantic_guard = SemanticGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)
    # Adversarial (L1/L2)
    adversarial_guard = AdversarialGuardrail(model, tokenizer, device, TS_PARSER, C_LANGUAGE, args)
    
    stats = {
        "TP": 0, "TN": 0, "FP": 0, "FN": 0, 
        "Total_Adv": 0, "Total_Benign": 0,
        "FP_Semantic": 0, "FP_Adversarial_Hard": 0, "FP_Adversarial_Grey": 0,
        "TP_Semantic": 0, "TP_Adversarial_Hard": 0, "TP_Adversarial_Grey": 0
    }
    
    fp_samples = []
    fn_samples = [] 

    print(f"[-] Strategy: Semantic First -> Adversarial Second")
    print(f"    Semantic Params -> Base Influence: {args.l3_base_influence}, Tolerance: {args.l3_surprise_tolerance}")
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, ncols=100, desc="Defending"):
            try: entry = json.loads(line)
            except: continue

            def run_defense_pipeline(code_snippet):
                code_to_check = code_snippet if code_snippet else ""
                adv_l1, adv_l2, clean_structure_code = adversarial_guard.detect(code_to_check)
                sem_detected, final_code, sem_debug = semantic_guard.detect(clean_structure_code)
                
                return {
                    "Semantic": sem_detected,         # semantic 結果
                    "Adversarial_Hard": adv_l1,       # L1 結果
                    "Adversarial_Grey": adv_l2,       # L2 結果
                    "debug": sem_debug,
                    "final_code": final_code          # 這是經過兩層清洗後的最終代碼
                }

            # --- Benign Test ---
            stats["Total_Benign"] += 1
            benign_code = entry.get("code") or ""
            res = run_defense_pipeline(benign_code)
            
            is_detected = res["Semantic"] or res["Adversarial_Hard"] or res["Adversarial_Grey"]
            
            if is_detected: 
                stats["FP"] += 1
                if res["Semantic"]: stats["FP_Semantic"] += 1
                if res["Adversarial_Hard"]: stats["FP_Adversarial_Hard"] += 1
                if res["Adversarial_Grey"]: stats["FP_Adversarial_Grey"] += 1
                
                triggered_debug = [d for d in res["debug"] if d['triggered']]
                if len(fp_samples) < 5 and triggered_debug:
                    fp_samples.append({
                        "id": stats["Total_Benign"],
                        "code": benign_code[:100],
                        "debug": triggered_debug
                    })
            else: 
                stats["TN"] += 1

            # --- Adversarial Test ---
            stats["Total_Adv"] += 1
            adv_code = entry.get("adv_code") or ""
            adv_res = run_defense_pipeline(adv_code)
            
            is_detected_adv = adv_res["Semantic"] or adv_res["Adversarial_Hard"] or adv_res["Adversarial_Grey"]
            
            if is_detected_adv: 
                stats["TP"] += 1
                if adv_res["Semantic"]: stats["TP_Semantic"] += 1
                if adv_res["Adversarial_Hard"]: stats["TP_Adversarial_Hard"] += 1
                if adv_res["Adversarial_Grey"]: stats["TP_Adversarial_Grey"] += 1
            else: 
                stats["FN"] += 1
                if len(fn_samples) < 5:
                    fn_samples.append({
                        "id": stats["Total_Adv"],
                        "code": adv_code[:100],
                        "debug": adv_res["debug"]
                    })
            
            entry["repaired_code"] = adv_res["final_code"]
            entry["defense_detected"] = is_detected_adv
            entry["layer_triggers"] = {
                "Semantic": adv_res["Semantic"],
                "Adversarial_Hard": adv_res["Adversarial_Hard"],
                "Adversarial_Grey": adv_res["Adversarial_Grey"]
            }
            out_f.write(json.dumps(entry) + "\n")

    # --- Metrics Calculation ---
    tp = stats["TP"]
    tn = stats["TN"]
    fp = stats["FP"]
    fn = stats["FN"]

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("My Defense framework Result")
    print(f"Precision:    {precision * 100:.2f}%")
    print(f"Recall:       {recall * 100:.2f}%")
    print(f"F1-Score:     {f1_score:.2f}")
    print(f"FPR:          {fpr * 100:.2f}%")

    print(f"False Positives Breakdown:")
    print(f"  Adversarial Guardrail (Hard): {stats['FP_Adversarial_Hard']}")
    print(f"  Adversarial Guardrail (Grey): {stats['FP_Adversarial_Grey']}")
    print(f"  Semantic Guardrail:           {stats['FP_Semantic']}")
    print("-" * 30)
    print(f"True Positives Breakdown:")
    print(f"  Adversarial Guardrail (Hard): {stats['TP_Adversarial_Hard']}")
    print(f"  Adversarial Guardrail (Grey): {stats['TP_Adversarial_Grey']}")
    print(f"  Semantic Guardrail:           {stats['TP_Semantic']}")
    
    eval_dir = "result/evaluations"
    eval_file = os.path.join(eval_dir, "f1_score_shadowcode.json")
    os.makedirs(eval_dir, exist_ok=True)

    current_run_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "parameters": {
            "th_comment_grey": args.th_comment_grey,
            "th_comment_hard": args.th_comment_hard,
            "th_string_hard": args.th_string_hard,
            "l3_base_influence": args.l3_base_influence,
            "l3_surprise_tolerance": args.l3_surprise_tolerance
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "fpr": round(fpr, 4)
        },
        "breakdown": {
            "false_positives": {
                "total": stats["FP"],
                "L1_hard": stats['FP_Adversarial_Hard'],
                "L2_grey": stats['FP_Adversarial_Grey'],
                "L3_semantic": stats['FP_Semantic']
            },
            "true_positives": {
                "total": stats["TP"],
                "L1_hard": stats['TP_Adversarial_Hard'],
                "L2_grey": stats['TP_Adversarial_Grey'],
                "L3_semantic": stats['TP_Semantic']
            }
        }
    }

    existing_data = []
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                existing_data = content if isinstance(content, list) else [content]
        except json.JSONDecodeError:
            existing_data = []

    existing_data.append(current_run_data)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n[-] Evaluation results saved to: {eval_file}")

if __name__ == "__main__":
    main()