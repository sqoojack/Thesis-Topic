
# python3 main_code/attack/Semantic_attack/compare_ASR.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_asr_chart():
    # 1. 定義數據
    data = {
        'Model': [
            'deepseek-r1:8b', 
            'deepseek-r1:32b', 
            'qwen2.5:7b', 
            'qwen3:14b', 
            'gemma3:27b', 
            'llama3:8b', 
            'mistral:7b', 
            'gpt-oss:20b', 
            'codellama:7b',
            'ministral-3:8b',
            'ministral-3:14b'
        ],
        'ASR': [
            18.1, 
            15.2, 
            18.7, 
            16.3, 
            7.4, 
            17.4, 
            19.5, 
            9.2, 
            21.6,
            14.5,
            20.8
        ]
    }

    # 2. 轉換為 DataFrame 並排序
    df = pd.DataFrame(data)
    df = df.sort_values('ASR', ascending=True)

    # 3. 設定圖表樣式
    plt.figure(figsize=(12, 8))
    
    # 設定顏色邏輯
    colors = []
    min_val = df['ASR'].min()
    max_val = df['ASR'].max()
    
    for val in df['ASR']:
        if val == min_val:
            colors.append('#2ca02c') # 綠色 (最低值)
        elif val == max_val:
            colors.append('#d62728') # 紅色 (最高值)
        else:
            colors.append('#1f77b4') # 藍色 (其他)

    # 4. 繪製圖表
    bars = plt.barh(df['Model'], df['ASR'], color=colors, height=0.6)

    # 5. 添加裝飾與標籤
    plt.title('Comparison of ASR by Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('ASR Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Model Name', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, max_val + 4)

    # 6. 添加數值標籤
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{width}%',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # plt.tight_layout()

    # 7. 儲存檔案 (不顯示)
    filename = 'compare.png'
    # dpi=300 確保圖片清晰度高
    # bbox_inches='tight' 確保邊緣文字不會被切掉
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # 關閉圖表釋放記憶體
    
    print(f"圖表已成功儲存為: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_asr_chart()