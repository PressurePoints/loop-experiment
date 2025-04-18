import re
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

id = [3476, 3358]
n = 8

def extract_datas_from_file(file_path, row_start_index, column_index):
    datas = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(row_start_index, row_start_index + 2): # 2 rows start from row_start_index
            line = lines[i].strip()
            parts = [p for p in line.split(' ') if p]
            datas.append(float(parts[column_index])) # column_index
    return datas

def prepare_data(file_path_head, file_names, expected_file_name):
    revenue1 = []
    revenue2 = []
    scores11 = []
    scores12 = []
    scores21 = []
    scores22 = []
    scores31 = []
    scores32 = []
    for file_name in file_names:
        file_path = file_path_head + file_name
        revenues = extract_datas_from_file(file_path, 3, 6)
        score1s = extract_datas_from_file(file_path, 10, 1)
        score2s = extract_datas_from_file(file_path, 10, 2)
        score3s = extract_datas_from_file(file_path, 10, 3)
        revenue1.append(revenues[0])
        revenue2.append(revenues[1])
        scores11.append(score1s[0])
        scores12.append(score1s[1])
        scores21.append(score2s[0])
        scores22.append(score2s[1])
        scores31.append(score3s[0])
        scores32.append(score3s[1])

    expected_revenues = extract_datas_from_file(file_path_head + expected_file_name, 3, 6)
    expected_revenue1 = expected_revenues[0]
    expected_revenue2 = expected_revenues[1]
    return revenue1, revenue2, scores11, scores12, scores21, scores22, scores31, scores32, expected_revenue1, expected_revenue2

def draw_ratio_change(data1, data2, have_expection, expected_data1, expected_data2, ylabel, title):
    ratios_advertiser1 = []
    for d1, d2 in zip(data1, data2):
        if d1 < 0 and d2 < 0:
            ratios_advertiser1.append(-1)
        elif d1 < 0 and d2 > 0:
            ratios_advertiser1.append(0)
        elif d1 > 0 and d2 < 0:
            ratios_advertiser1.append(1)
        else:
            ratios_advertiser1.append(d1 / (d1 + d2))
    ratios_advertiser2 = [1 - ratio for ratio in ratios_advertiser1]
    print(ratios_advertiser1)
    print("\n")

    round_numbers = list(range(1, n))
    plt.figure(figsize=(6, 4))

    # 绘制堆叠柱状图
    bar_width = 0.6
    bars1 = plt.bar(round_numbers, ratios_advertiser1, bar_width, label='Advertiser 1: ' + str(id[0]))
    bars2 = plt.bar(round_numbers, ratios_advertiser2, bar_width, bottom=ratios_advertiser1, label='Advertiser 2: ' + str(id[1]))
    if have_expection:
        # 添加预期收益占比的虚线
        expected_ratio_advertiser1 = 1.0 * expected_data1 / (expected_data1 + expected_data2)
        expected_ratios_advertiser1 = [expected_ratio_advertiser1] * len(round_numbers)
        plt.plot(round_numbers, expected_ratios_advertiser1, linestyle='--', color='red', linewidth=2, label=f'广告主1在V2流程最终的收益占比 ({expected_ratio_advertiser1*100:.0f}%)')
    # 添加标签和标题
    plt.xlabel('循环轮次')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(round_numbers)
    plt.legend(loc='upper right')
    plt.tight_layout()


def main():
    file_path_head = 'loop-experiment/output/lr-tb/'
    file_path_head += (str(id[0]) + '-' + str(id[1]) + '/')

    file_base_name = str(id[0]) + '_' + str(id[1])
    expected_file_name = file_base_name + 'V2.txt'
    file_names = []
    for i in range(1, n):
        file_name = '[' + str(i) + ']' + file_base_name + 'V1.txt'
        file_names.append(file_name)
    
    revenue1, revenue2, scores11, scores12, scores21, scores22, scores31, scores32, expected_revenue1, expected_revenue2 = prepare_data(file_path_head, file_names, expected_file_name)
    draw_ratio_change(revenue1, revenue2, True, expected_revenue1, expected_revenue2, 'Revenue 占比', '不同循环轮次中广告主1和广告主2的收益占比')
    draw_ratio_change(scores11, scores12, False, 0, 0, 'Score 1 占比', '不同循环轮次中广告主1和广告主2的Score 1占比')
    draw_ratio_change(scores21, scores22, False, 0, 0, 'Score 2 占比', '不同循环轮次中广告主1和广告主2的Score 2占比')
    draw_ratio_change(scores31, scores32, False, 0, 0, 'Score 3 占比', '不同循环轮次中广告主1和广告主2的Score 3占比')
    plt.show()

if __name__ == '__main__':
    main()