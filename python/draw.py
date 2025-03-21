import re
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

id = [2259, 2261]

revenue1 = [1313721, 5718643, 1137590, 847592]
total_revenue = [1717361, 11505736, 4423593, 4708155]
ratios_advertiser1 = [1.0 * r / t for r, t in zip(revenue1, total_revenue)]
ratios_advertiser2 = [1 - ratio for ratio in ratios_advertiser1]
expected_ratio_advertiser1 = 1.0 * 2488483 / 5502063

size1 = [203889, 146115, 139644, 100099]
size2 = [182898, 161655, 163559, 135718]
size1_ratio = [1.0 * s1 / (s1 + s2) for s1, s2 in zip(size1, size2)]
size2_ratio = [1 - ratio for ratio in size1_ratio]
expected_size1_ratio = 1.0 * 250551 / (206296 + 250551)

def main():
    DrawRevenueRatioChange()
    DrawSizeRatioChange()
    plt.show()

def DrawRevenueRatioChange():
    round_numbers = list(range(1, len(ratios_advertiser1) + 1))

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

def DrawSizeRatioChange():
    round_numbers = list(range(1, len(ratios_advertiser1) + 1))

    plt.figure(figsize=(6, 4))

    # 绘制堆叠柱状图
    bar_width = 0.6
    bars1 = plt.bar(round_numbers, size1_ratio, bar_width, label='Advertiser 1: ' + str(id[0]))
    bars2 = plt.bar(round_numbers, size2_ratio, bar_width, bottom=size1_ratio, label='Advertiser 2: ' + str(id[1]))
    # 添加预期收益占比的虚线
    expected_size1_ratios = [expected_size1_ratio] * len(round_numbers)
    plt.plot(round_numbers, expected_size1_ratios, linestyle='--', color='red', linewidth=2, label=f'每份数据中广告主1的size期望占比 ({expected_ratio_advertiser1*100:.0f}%)')
    # 添加标签和标题
    plt.xlabel('循环轮次')
    plt.ylabel('竞胜数据的size占比')
    plt.title('不同循环轮次中广告主1和广告主2的竞胜数据size占比')
    plt.xticks(round_numbers)
    plt.legend(loc='upper right')

    plt.tight_layout()

if __name__ == '__main__':
    main()