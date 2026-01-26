import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 讀取資料
df = pd.read_csv('shadowcode_results.csv')

# 2. 資料清理：移除不需要的 perturbation 欄位，並去除空值
df = df[['case_id', 'language', 'dataset', 'asr']].dropna()

# 為了讓圖表更好看，我們將 case_id 排序
df = df.sort_values(by=['case_id', 'asr'])

# 3. 設定繪圖風格
sns.set_theme(style="whitegrid")

# 4. 繪製圖表
# 我們以 case_id 為 X 軸，asr 為 Y 軸，並用顏色 (hue) 區分 dataset 或 language
plt.figure(figsize=(14, 8))
plot = sns.barplot(
    data=df, 
    x='case_id', 
    y='asr', 
    hue='dataset',  # 你也可以換成 'language' 看看不同的比較維度
    palette='viridis'
)

# 5. 優化圖表細節
plt.title('Comparison of ASR across Case IDs and Datasets', fontsize=16)
plt.xlabel('Case ID', fontsize=12)
plt.ylabel('ASR (%)', fontsize=12)
plt.xticks(rotation=45) # 旋轉 X 軸標籤避免重疊
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 6. 儲存與顯示
plt.savefig('comparison_chart.png')
# plt.show() # 在本地環境執行時可取消註解