# logistic regression + truthful bidding

from dataset import Dataset
from ctr_estimator import LrModel

import sys
import os

train_round = 20
id = [2259, 2261]
n = 5
data_folder = "make-ipinyou-data/"

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

# 1458 3386
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

    camp_vs = get_camp_vs(datas_lists)
    for i in range(len(id)):
        print("camp_v%d: %d" % (i, camp_vs[i]))

    V1(datas_lists, camp_vs, file_base_name, output_path)
    V2(datas_lists, camp_vs, file_base_name, output_path)
    V3(datas_lists, camp_vs, file_base_name, output_path)

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
    train_datasets = [Dataset(data_lists[i][0], id[i]) for i in range(len(id))] # first part
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