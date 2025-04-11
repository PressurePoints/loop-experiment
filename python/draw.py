import re
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

id = [3476, 3358]
n = 8

def extract_revenues_from_file(file_path):
    revenues = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(3, 5): # 4th and 5th lines
            line = lines[i].strip()
            parts = [p for p in line.split(' ') if p]
            revenues.append(int(parts[6])) # 7th column
    return revenues

def extract_winsizes_from_file(file_path):
    winsizes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(3, 5): # 4th and 5th lines
            line = lines[i].strip()
            parts = [p for p in line.split(' ') if p]
            winsizes.append(int(parts[7])) # 8th column
    return winsizes    

def main():
    file_path_head = 'loop-experiment/output/lr-tb/'
    file_path_head += (str(id[0]) + '-' + str(id[1]) + '/')

    file_base_name = str(id[0]) + '_' + str(id[1])
    expected_file_name = file_base_name + 'V2.txt'
    file_names = []
    for i in range(1, n):
        file_name = '[' + str(i) + ']' + file_base_name + 'V1.txt'
        file_names.append(file_name)

    DrawRevenueRatioChange(file_path_head, file_names, expected_file_name)
    DrawSizeRatioChange(file_path_head, file_names, expected_file_name)
    plt.show()

def DrawRevenueRatioChange(file_path_head, file_names, expected_file_name):
    revenue1 = []
    revenue2 = []
    for file_name in file_names:
        file_path = file_path_head + file_name
        revenues = extract_revenues_from_file(file_path)
        revenue1.append(revenues[0])
        revenue2.append(revenues[1])
    expected_revenues = extract_revenues_from_file(file_path_head + expected_file_name)
    expected_revenue1 = expected_revenues[0]
    expected_revenue2 = expected_revenues[1]
    print(revenue1)
    print(revenue2)
    print(expected_revenue1)
    print(expected_revenue2)

    ratios_advertiser1 = []
    for r1, r2 in zip(revenue1, revenue2):
        if r1 < 0 and r2 < 0:
            ratios_advertiser1.append(-1)
        elif r1 < 0 and r2 > 0:
            ratios_advertiser1.append(0)
        elif r1 > 0 and r2 < 0:
            ratios_advertiser1.append(1)
        else:
            ratios_advertiser1.append(r1 / (r1 + r2))
    ratios_advertiser2 = [1 - ratio for ratio in ratios_advertiser1]
    expected_ratio_advertiser1 = 1.0 * expected_revenue1 / (expected_revenue1 + expected_revenue2)
    print(ratios_advertiser1)
    print(expected_ratio_advertiser1)
    print("\n")

    round_numbers = list(range(1, n))

    plt.figure(figsize=(6, 4))

    # 绘制堆叠柱状图
    bar_width = 0.6
    bars1 = plt.bar(round_numbers, ratios_advertiser1, bar_width, label='Advertiser 1: ' + str(id[0]))
    bars2 = plt.bar(round_numbers, ratios_advertiser2, bar_width, bottom=ratios_advertiser1, label='Advertiser 2: ' + str(id[1]))
    # 添加预期收益占比的虚线
    expected_ratios_advertiser1 = [expected_ratio_advertiser1] * len(round_numbers)
    plt.plot(round_numbers, expected_ratios_advertiser1, linestyle='--', color='red', linewidth=2, label=f'广告主1在V2流程最终的收益占比 ({expected_ratio_advertiser1*100:.0f}%)')
    # 添加标签和标题
    plt.xlabel('循环轮次')
    plt.ylabel('Revenue 占比')
    plt.title('不同循环轮次中广告主1和广告主2的收益占比')
    plt.xticks(round_numbers)
    plt.legend(loc='upper right')

    plt.tight_layout()

def DrawSizeRatioChange(file_path_head, file_names, expected_file_name):
    size1 = []
    size2 = []
    for file_name in file_names:
        file_path = file_path_head + file_name
        sizes = extract_winsizes_from_file(file_path)
        size1.append(sizes[0])
        size2.append(sizes[1])
    expected_sizes = extract_winsizes_from_file(file_path_head + expected_file_name)
    expected_size1 = expected_sizes[0]
    expected_size2 = expected_sizes[1]
    print(size1)
    print(size2)
    print(expected_size1)
    print(expected_size2)

    size1_ratio = [1.0 * s1 / (s1 + s2) for s1, s2 in zip(size1, size2)]
    size2_ratio = [1 - ratio for ratio in size1_ratio]
    expected_size1_ratio = 1.0 * expected_size1 / (expected_size1 + expected_size2)
    print(size1_ratio)
    print(expected_size1_ratio)
    print("\n")

    round_numbers = list(range(1, n))

    plt.figure(figsize=(6, 4))

    # 绘制堆叠柱状图
    bar_width = 0.6
    bars1 = plt.bar(round_numbers, size1_ratio, bar_width, label='Advertiser 1: ' + str(id[0]))
    bars2 = plt.bar(round_numbers, size2_ratio, bar_width, bottom=size1_ratio, label='Advertiser 2: ' + str(id[1]))
    # 添加预期收益占比的虚线
    expected_size1_ratios = [expected_size1_ratio] * len(round_numbers)
    plt.plot(round_numbers, expected_size1_ratios, linestyle='--', color='red', linewidth=2, label=f'每份数据中广告主1的size期望占比 ({expected_size1_ratio*100:.0f}%)')
    # 添加标签和标题
    plt.xlabel('循环轮次')
    plt.ylabel('竞胜数据的size占比')
    plt.title('不同循环轮次中广告主1和广告主2的竞胜数据size占比')
    plt.xticks(round_numbers)
    plt.legend(loc='upper right')

    plt.tight_layout()

if __name__ == '__main__':
    main()