
# python main_code/attack/Syntactic_Structure_attack/compare_ASR.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def create_asr_chart():
    # ==========================================
    # 0. 路徑設定 (Path Setup)
    # ==========================================
    # 取得當前腳本所在的絕對路徑
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 設定 result 資料夾路徑
    RESULT_DIR = os.path.join(CURRENT_DIR, "result")
    
    # 輸入與輸出檔案的完整路徑
    json_filename = os.path.join(RESULT_DIR, "ASR_summary.json")
    output_filename = os.path.join(RESULT_DIR, "compare_ASR.png")

    # ==========================================
    # 1. 讀取與處理數據
    # ==========================================
    print(f"正在讀取數據來源: {json_filename}")

    if not os.path.exists(json_filename):
        print(f"錯誤: 找不到檔案 {json_filename}")
        print(f"請確認該檔案是否存在於: {RESULT_DIR}")
        return

    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"讀取 JSON 失敗: {e}")
        return

    models = []
    asr_values = []

    for entry in raw_data:
        # 獲取 Model 名稱
        filename = entry.get('input_filename', '')
        model_name = filename.replace('_response.csv', '')
        
        # 獲取 ASR 數值
        asr_str = entry.get('ASR (Attack Success Rate)', '0%')
        try:
            asr_val = float(asr_str.replace('%', '').strip())
        except ValueError:
            asr_val = 0.0

        models.append(model_name)
        asr_values.append(asr_val)

    # 建構 DataFrame
    df = pd.DataFrame({
        'Model': models,
        'ASR': asr_values
    })

    # ==========================================
    # 2. 排序與繪圖準備
    # ==========================================
    df = df.sort_values('ASR', ascending=True)

    plt.figure(figsize=(12, 8))
    
    colors = []
    if not df.empty:
        min_val = df['ASR'].min()
        max_val = df['ASR'].max()
        
        for val in df['ASR']:
            if val == min_val:
                colors.append('#2ca02c') 
            elif val == max_val:
                colors.append('#d62728') 
            else:
                colors.append('#1f77b4') 
    else:
        colors = '#1f77b4'

    # ==========================================
    # 3. 繪製圖表
    # ==========================================
    bars = plt.barh(df['Model'], df['ASR'], color=colors, height=0.6)

    plt.title('Comparison of ASR by Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('ASR Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Model Name', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    limit_max = max_val + 5 if not df.empty else 100
    plt.xlim(0, limit_max)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,                  
            bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}%',              
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # ==========================================
    # 4. 儲存檔案
    # ==========================================
    # 使用前面設定好的絕對路徑 output_filename
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分析完成！圖表已儲存為: {output_filename}")

if __name__ == "__main__":
    create_asr_chart()