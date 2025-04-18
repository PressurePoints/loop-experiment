from dataset import Dataset
from ctr_estimator import LrModel

import sys
import os
import random
import numpy as np

data_folder = "make-ipinyou-data/"
output_folder = "loop-experiment/output/quality-revenue-relationship/"
id = 3358
train_round = 20
N = 30

def get_datas(data_path):
    if not os.path.exists(data_path):
        print("ERROR: file not exist. " + data_path)
        exit(-1)
    fi = open(data_path, 'r')
    datas = [] # all yzx data
    for line in fi:
        data = []
        for d in line.replace(':1','').split():
            data.append(int(d))
        datas.append(data)
    fi.close()
    return datas

def prepare_dataset():
    train_path = data_folder + str(id) + "/train.yzx.txt"
    test_path = data_folder + str(id) + "/test.yzx.txt"

    complete_datas_train = get_datas(train_path)
    datas_test = get_datas(test_path)
    complete_train_dataset = Dataset(complete_datas_train, id)
    cut_index = int(len(datas_test) / 2)
    test_dataset1 = Dataset(datas_test[:cut_index], id)
    test_dataset2 = Dataset(datas_test[cut_index:], id)
    camp_v = complete_train_dataset.statistics['ecpc']
    print("camp_v: ", camp_v)
    print("complete_train_dataset size: ", complete_train_dataset.statistics['size'])
    print("test_dataset1 size: ", test_dataset1.statistics['size'])
    print("test_dataset2 size: ", test_dataset2.statistics['size'])
    return camp_v, complete_datas_train, complete_train_dataset, test_dataset1, test_dataset2

def first_step(camp_v, train_dataset, test_dataset1):
    ctr_estimator = LrModel([train_dataset], [test_dataset1], [id], [camp_v])
    initial_performance = ctr_estimator.init_performances[0]
    print("Initial performance: ", initial_performance)
    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        for performance in this_round_log['performances']:
            print("Round " + str(i+1) + "\t" + str(performance))
    print("Train done.")

    weight = ctr_estimator.get_best_test_log()['weight']
    performance = ctr_estimator.get_best_test_log()['performances'][0]
    winning_dataset = ctr_estimator.winning_datasets[0]
    return initial_performance, weight, performance, winning_dataset

def second_step(camp_v, weight, winning_dataset, test_dataset2):
    ctr_estimator = LrModel([winning_dataset], [test_dataset2], [id], [camp_v], weight)
    initial_performance = ctr_estimator.init_performances[0]
    print("Initial performance: ", initial_performance)
    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        for performance in this_round_log['performances']:
            print("Round " + str(i+1) + "\t" + str(performance))
    print("Train done.") 
    final_performance = ctr_estimator.get_best_test_log()['performances'][0]
    return initial_performance, final_performance  

def main():
    camp_v, complete_datas_train, complete_train_dataset, test_dataset1, test_dataset2 = prepare_dataset()
    train_size = int(complete_train_dataset.statistics['size'] * (2 / 3))

    output_path = output_folder + str(id) + ".txt"
    os.makedirs(output_folder, exist_ok=True)
    fo = open(output_path, 'w')     # a for append, w for overwrite
    headers = ['Seed', 'Revenue', 'Train score1', 'Train score2', 'Train score3', 'Winning score1', 'Winning score2', 'Winning score3']
    header_line = '{:<8} {:<8} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}'.format(*headers)
    fo.write(header_line + '\n')

    for i in range(0, N):
        print("\n" + "Count: " + str(i+1) + "\n")
        seed = i * 10
        random.seed(seed)
        random.shuffle(complete_datas_train)
        datas_train = complete_datas_train[:train_size]
        train_dataset = Dataset(datas_train, id)
        train_dataset.output_statistics()
        test_dataset1.output_statistics()
        test_dataset2.output_statistics()

        initial_performance, weight, performance, winning_dataset = first_step(camp_v, train_dataset, test_dataset1)
        initial_performance2, final_performance = second_step(camp_v, weight, winning_dataset, test_dataset2)

        revenue = performance['revenue']
        train_quality_score1 = score_quality_with_positive_ratio(train_dataset)
        train_quality_score2 = score_quality_with_positive_feature(train_dataset, complete_train_dataset)
        train_quality_score3 = score_quality_with_shapley_value(initial_performance, performance, train_size)
        winning_quality_score1 = score_quality_with_positive_ratio(winning_dataset)
        winning_quality_score2 = score_quality_with_positive_feature(winning_dataset, test_dataset1)
        winning_quality_score3 = score_quality_with_shapley_value(initial_performance2, final_performance, winning_dataset.statistics['size'])
        print("Revenue: ", revenue)
        print("Train quality score1: ", train_quality_score1)
        print("Train quality score2: ", train_quality_score2)
        print("Train quality score3: ", train_quality_score3)
        print("Winning quality score1: ", winning_quality_score1)
        print("Winning quality score2: ", winning_quality_score2)
        print("Winning quality score3: ", winning_quality_score3)

        line = '{:<8} {:<8} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}'.format(seed, revenue, train_quality_score1, train_quality_score2, train_quality_score3, winning_quality_score1, winning_quality_score2, winning_quality_score3)
        fo.write(line + '\n')
    fo.close()

def score_quality_with_positive_ratio(test_dataset):
    ctr = test_dataset.statistics['ctr']
    log_ctr = np.log(ctr + 1e-6)  
    min_log = np.log(1e-6)  # 理论最小值
    max_log = np.log(0.002)   # 选择 0.002 为上限
    return (log_ctr - min_log) / (max_log - min_log)

def score_quality_with_positive_feature(test_dataset, all_dataset):
    test_dataset_positive_feature_num = test_dataset.statistics['positive_feature_num']
    all_dataset_positive_feature_num = all_dataset.statistics['positive_feature_num']
    return test_dataset_positive_feature_num / all_dataset_positive_feature_num

def score_quality_with_shapley_value(initial_performance, final_performance, train_dataset_size):
    if initial_performance == None:
        initial_revenue = 0
    else:
        initial_revenue = initial_performance['revenue']
    final_revenue = final_performance['revenue']
    return (final_revenue - initial_revenue) / train_dataset_size

if __name__ == '__main__':
    main()