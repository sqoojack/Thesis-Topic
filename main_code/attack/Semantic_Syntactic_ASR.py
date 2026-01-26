# python main_code/attack/Semantic_Syntactic_ASR.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 數據定義
# 第一張圖 (Syntactic Structure Attack)
syntactic_data = {
    "deepseek-r1_8b": 10.0,
    "deepseek-r1_32b": 35.56,
    "qwen2.5_7b": 51.11,
    "qwen3_14b": 44.44,
    "gemma3_27b": 56.67,
    "llama3_8b": 5.56,
    "mistral_7b": 28.89,
    "gpt-oss_20b": 36.67,
    "codellama_7b": 33.33,
    "ministral-3_8b": 37.78,
    "ministral-3_14b": 36.67,
}

# 第二張圖 (Semantic Attack)
semantic_data = {
    "deepseek-r1_8b": 18.1,
    "deepseek-r1_32b": 15.2,
    "qwen2.5_7b": 18.7,
    "qwen3_14b": 16.3,
    "gemma3_27b": 7.4,
    "llama3_8b": 17.4,
    "mistral_7b": 19.5,
    "gpt-oss_20b": 9.2,
    "codellama_7b": 21.6,
    "ministral-3_8b": 14.5,
    "ministral-3_14b": 20.8,
}

# 2. 數據整合與對齊
all_models = sorted(list(set(syntactic_data.keys()) | set(semantic_data.keys())))

df = pd.DataFrame({
    'Model': all_models,
    'Syntactic_ASR': [syntactic_data.get(m, 0) for m in all_models],
    'Semantic_ASR': [semantic_data.get(m, 0) for m in all_models]
})

# 依據數值排序
df = df.sort_values(by=['Syntactic_ASR', 'Semantic_ASR'], ascending=True)

# 3. 繪圖設定
models = df['Model']
syntactic = df['Syntactic_ASR']
semantic = df['Semantic_ASR']

y = np.arange(len(models))
height = 0.35  # 柱狀圖寬度

fig, ax = plt.subplots(figsize=(12, 10))

# 繪製分組柱狀圖
rects1 = ax.barh(y + height/2, syntactic, height, label='Syntactic Structure Attack ASR', color='#d62728')
rects2 = ax.barh(y - height/2, semantic, height, label='Semantic Attack ASR', color='#1f77b4')

# 4. 圖表修飾
ax.set_xlabel('ASR Percentage (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Model Name', fontweight='bold', fontsize=12)
ax.set_title('Comparison of ASR: Syntactic vs Semantic Attack', fontweight='bold', fontsize=16)
ax.set_yticks(y)
ax.set_yticklabels(models)
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.7)

# 自動標註數值
def autolabel(rects):
    for rect in rects:
        width = rect.get_width()
        if width > 0:
            ax.annotate(f'{width}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# 5. 存檔 (不使用 plt.show)
plt.savefig('merged_asr_comparison.png')
df.to_csv('merged_asr_data.csv', index=False)