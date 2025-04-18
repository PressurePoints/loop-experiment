# 根据winningprice分位数划分数据子集 价格敏感样本筛选
price_q75 = np.percentile(data['z'], 75)
high_price_data = data[data['z'] > price_q75]  # 高价区间样本

# 稀疏特征覆盖度检查 特征覆盖度和标签置信度
统计每个特征字段在子集中的出现频率（如27:1是否在所有子集均匀分布）
论文："Feature Coverage for Better Generalization" (ICML 2019 Workshop)

1. 分布相似性测试
KL散度比较特征分布：scipy.stats.entropy(p_subset, q_fullset)
特别关注高频特征（如30:1~43:1）的分布偏移

使用sklearn.metrics.mutual_info_score比较子集与全集的特征-标签关系

# 预算感知评估
定义质量指标：(子集总点击量 / 子集总winningprice)
论文："Budget Pacing for Online Ad Campaigns" (WWW 2015)

# 验证winningprice分布与全集的KS检验（scipy.stats.kstest）

# 计算特征共现差异（例如27:1和30:1的共现频率变化）
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(one_hot_data, min_support=0.1)


# 数据沙普利值（Shapley Value）
https://www.isclab.org.cn/wp-content/uploads/2025/02/%E6%95%B0%E6%8D%AE%E6%A0%B7%E6%9C%AC%E7%9A%84%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0%E6%96%B9%E6%B3%95-%E9%A9%AC%E8%A5%BF%E6%B4%8B.pdf



A survey on dataset quality in machine learning
# 点击率大小（拿到的正样本比例） 从原比例到100%（边界符合，也符合收入高低情况）  Dataset imbalance metrics    inter-class imbalance 
百分比 0到1 问题 大部分比例很低
对数尺度转换  参考论文："A Logarithmic Loss for CTR Prediction" (RecSys 2020) 证明对数尺度更符合CTR预估的误差敏感度 ？
    # 加平滑避免log(0)
    log_ctr = np.log(ctr + 1e-6)  
    # 线性映射到[0,1]
    min_log = np.log(1e-6)  # 理论最小值
    max_log = np.log(0.5)   # 设定合理上限(如50% CTR)
    return (log_ctr - min_log) / (max_log - min_log)
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
# 收入的提高值（用新的训练数据 vs 不用 ） / 训练集大小 代表 训练数据质量    训练集整体的 Shapley Value
# 正样本中特征覆盖的比例（子集中正样本包含的特征数/全集中正样本包含的特征数）    4.1.3. Data testing   Intra-class imbalance
# KL散度比较特征分布：scipy.stats.entropy(p_subset, q_fullset) 训练集的特征分布和测试集的特征分布差距   4.1.3. Data testing   data distribution consistency    差距越小质量越高？与1矛盾
对于one-hot编码的稀疏特征（如数据中的27:1），直接计算KL散度可能失效，改为计算特征出现频率的JS散度
    train_feat: 训练集特征值数组
    from scipy.spatial.distance import jensenshannon
    freq_train = np.mean(train_X, axis=0)  # 各特征出现频率
    freq_test = np.mean(test_X, axis=0)
    js_distance = jensenshannon(freq_train, freq_test)