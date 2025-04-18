import os
import numpy as np
import pandas as pd

def extract_ctr_from_file(file_path):
    ctrs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(3, 5): # 4th and 5th lines
            line = lines[i].strip()
            parts = [p for p in line.split(' ') if p]
            ctrs.append(float(parts[4]))
    return ctrs[0], ctrs[1]

def output_metrics(ctrs):
    stats = {
        "样本量": len(ctrs),
        "平均值": np.mean(ctrs),
        "中位数": np.median(ctrs),
        "最小值": np.min(ctrs),
        "最大值": np.max(ctrs),
        "标准差": np.std(ctrs),
        "25%分位数": np.percentile(ctrs, 25),
        "75%分位数": np.percentile(ctrs, 75),
    }
    # 转换为DataFrame输出更美观
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['值'])
    print(stats_df)

if __name__ == "__main__":
    ctrs = []
    folder_path = "loop-experiment/output/lr-tb"  
    for item in os.listdir(folder_path):
        if item != "2259-2261-cut-200":
            for filename in os.listdir(os.path.join(folder_path, item)):
                ctr1, ctr2 = extract_ctr_from_file(os.path.join(folder_path, item, filename))
                ctrs.append(ctr1)
                ctrs.append(ctr2)
    
    output_metrics(ctrs)

# 输出结果如下：
# 选择 0.002 为上限
'''
样本量     144.000000
平均值       0.000771
中位数       0.000717
最小值       0.000137
最大值       0.001814
标准差       0.000394
25%分位数    0.000435
75%分位数    0.001095
'''