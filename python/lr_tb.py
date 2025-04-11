# logistic regression + truthful bidding

from dataset import Dataset
from ctr_estimator import LrModel

import sys
import os
import random

# ad_id     total_bids
# 1458      3697694
# 3386      3393223
# 3427      3130560
# 3476      2494208
# 3358      2043032
# 2821      1984525
# 2259      1252753
# 2261      1031479
# 2997      468500
train_round = 20
id = [3476, 3358]
n = 8
data_folder = "make-ipinyou-data/"

default_order = False
V1_index_order = [0, 2, 3, 6, 7, 4, 5, 1]
# [0, 3, 6, 7, 4, 5, 2, 1]
# [0, 6, 7, 3, 2, 4, 1, 5]
# [0, 3, 2, 6, 7, 4, 5, 1]
# [0, 6, 7, 4, 5, 1, 3, 2]
# [0, 2, 3, 6, 7, 4, 5, 1]
def split_data(data_path):
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
    k, m = divmod(len(datas), n)
    datas_list = [datas[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    return datas_list

# 2259 2261
def main():
    file_base_name = str(id[0])
    if len(id) == 2:
        file_base_name = str(id[0]) + "_" + str(id[1])

    output_path = "loop-experiment/output/lr-tb/"
    output_path += str(id[0])
    if len(id) == 2:
        output_path += "-" + str(id[1])
    output_path += "/"

    data_paths = [data_folder + str(id[i]) + "/all.yzx.txt" for i in range(len(id))]
    print("Begin spliting data ...")
    datas_lists = [split_data(data_path) for data_path in data_paths]
    print("Data splited.")

    if not default_order:
        for i in range(len(id)):
            temp = datas_lists[i]
            for j in range(0, n):
                datas_lists[i][j] = temp[V1_index_order[j]]
        print("Index order changed.")

    camp_vs = get_camp_vs(datas_lists)
    for i in range(len(id)):
        print("camp_v%d: %d" % (i, camp_vs[i]))

    # output_path = output_path[:-1]
    # output_path += "-cut-200/"
    # random.seed(200)
    # cut_datas(datas_lists)
    # test_data_similarity(datas_lists)
    V1(datas_lists, camp_vs, file_base_name, output_path)
    V2(datas_lists, camp_vs, file_base_name, output_path)
    V3(datas_lists, camp_vs, file_base_name, output_path)

def cut_datas(datas_lists):
    # for every part of data, cut the first advertiser's data to let it be the same size, cost_sum, clk_sum as the second advertiser's data
    if len(datas_lists) != 2:
        return
    for i in range(0, n):
        # cut datas_lists[0][i] to the same as datas_lists[1][i]
        dataset1 = Dataset(datas_lists[1][i], id[1])
        target_size = dataset1.statistics['size']
        target_cost_sum = dataset1.statistics['cost_sum']
        target_clk_sum = dataset1.statistics['clk_sum']
        
        data0 = datas_lists[0][i]
        random.shuffle(data0)
        
        sampled_data = []
        current_size = 0
        current_cost_sum = 0
        current_clk_sum = 0
        for data in data0:
            if current_size >= target_size and current_cost_sum >= target_cost_sum and current_clk_sum >= target_clk_sum:
                break
            sampled_data.append(data)
            current_size += 1
            current_cost_sum += data[1]
            current_clk_sum += data[0]
        datas_lists[0][i] = sampled_data

def test_data_similarity(datas_lists):
    for i in range(0, n):
        print(str(i + 1) + "th part: ")
        for j in range(len(datas_lists)):
            print(str(j + 1) + "th advertiser's data info:")
            temp = Dataset(datas_lists[j][i], id[j])
            temp.output_statistics()

def get_camp_vs(datas_lists):
    camp_vs = []
    for datas_list in datas_lists:
        all_train_data = [item for sublist in datas_list[0 : n-1] for item in sublist] # n-1 parts for training
        train_dataset = Dataset(all_train_data, id[0])
        camp_v = train_dataset.statistics['ecpc']
        camp_vs.append(camp_v)

    return camp_vs

def V1(data_lists, camp_vs, file_base_name, output_path):
    #----------------- data -----------------
    train_datasets = [Dataset(data_lists[i][0], id[i]) for i in range(len(id))] # first part, also 0th part
    for train_dataset in train_datasets:
        train_dataset.shuffle()
    weight = None
    #----------------- train -----------------
    print("Begin training ...")
    for i in range(1, n):
        print("Turn " + str(i) + " ...")
        test_datasets = [Dataset(data_lists[j][i], id[j]) for j in range(len(id))] # ith part
        ctr_estimator = LrModel(train_datasets, test_datasets, id, camp_vs, weight)
        ctr_estimator.output_data_info()

        for j in range(0, train_round):
            ctr_estimator.train()
            ctr_estimator.test()
            this_round_log = ctr_estimator.get_last_test_log()
            for performance in this_round_log['performances']:
                print("Round " + str(j+1) + "\t" + str(performance))
        
        weight = ctr_estimator.get_best_test_log()['weight']
        train_datasets = ctr_estimator.winning_datasets
        for train_dataset in train_datasets:
            train_dataset.shuffle()
        #----------------- output -----------------
        print("Begin log output in this turn ...")
        log_file = "[" + str(i) + "]" + file_base_name + "V1.txt"
        os.makedirs(output_path, exist_ok=True)
        log_output_path = output_path + log_file
        ctr_estimator.output_log(log_output_path)

def V2(data_lists, camp_vs, file_base_name, output_path):
    #----------------- data -----------------
    print("Begin loading data ...")
    train_datasets = []
    test_datasets = []
    for i in range(len(id)):
        all_train_data = [item for sublist in data_lists[i][0 : n-1] for item in sublist] # n-1 parts for training
        train_dataset = Dataset(all_train_data, id[i])
        train_dataset.shuffle()
        test_dataset = Dataset(data_lists[i][n-1], id[i]) # n-1 th part, final part

        train_dataset.output_statistics()
        test_dataset.output_statistics()

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    print("Data loaded.")
    #----------------- train -----------------    
    ctr_estimator = LrModel(train_datasets, test_datasets, id, camp_vs)
    ctr_estimator.output_info()
    ctr_estimator.output_data_info()

    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        for performance in this_round_log['performances']:
            print("Round " + str(i+1) + "\t" + str(performance))
    print("Train done.")
    #----------------- output -----------------
    log_file = file_base_name + "V2.txt"
    os.makedirs(output_path, exist_ok=True)
    log_output_path = output_path + log_file
    print("Begin log output ...")
    ctr_estimator.output_log(log_output_path)
    print("Log output done.")

    # print("Begin weight output ...")
    # weight_path = str(id1) + "_" + str(id2) + "_" + "best_weight" \
    #             + "_" + str(ctr_estimator.lr_alpha) + "_" + str(ctr_estimator.budget_prop) \
    #             + ".weight"
    # best_test_log = ctr_estimator.get_best_test_log()
    # ctr_estimator.output_weight(best_test_log['weight'], "../output/" + weight_path)
    # print("Weight output done.")

def V3(data_lists, camp_vs, file_base_name, output_path):
    #----------------- data -----------------
    print("Begin loading data ...")
    train_datasets = []
    test_datasets = []
    for i in range(len(id)):
        train_dataset = Dataset(data_lists[i][0], id[i]) # first part
        train_dataset.shuffle()
        test_dataset = Dataset(data_lists[i][n-1], id[i]) # nth part

        train_dataset.output_statistics()
        test_dataset.output_statistics()

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    print("Data loaded.")
    #----------------- train -----------------    
    ctr_estimator = LrModel(train_datasets, test_datasets, id, camp_vs)
    ctr_estimator.output_info()
    ctr_estimator.output_data_info()

    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        for performance in this_round_log['performances']:
            print("Round " + str(i+1) + "\t" + str(performance))
    print("Train done.")
    #----------------- output -----------------
    log_file = file_base_name + "V3.txt"
    os.makedirs(output_path, exist_ok=True)
    log_output_path = output_path + log_file
    print("Begin log output ...")
    ctr_estimator.output_log(log_output_path)
    print("Log output done.")

    # print("Begin weight output ...")
    # weight_path = str(id1) + "_" + str(id2) + "_" + "best_weight" \
    #             + "_" + str(ctr_estimator.lr_alpha) + "_" + str(ctr_estimator.budget_prop) \
    #             + ".weight"
    # best_test_log = ctr_estimator.get_best_test_log()
    # ctr_estimator.output_weight(best_test_log['weight'], "../output/" + weight_path)
    # print("Weight output done.")


if __name__ == '__main__':
    main()