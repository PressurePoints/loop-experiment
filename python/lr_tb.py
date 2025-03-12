# logistic regression + truthful bidding

from dataset import Dataset
from ctr_estimator import LrModel

import sys
import os

train_round = 20
id = [2259, 2261]
n = 5
data_folder = "make-ipinyou-data/"
output_path = "loop-experiment/output/lr-tb/"

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
    data_paths = [data_folder + str(id[i]) + "/all.yzx.txt" for i in range(2)]
    print("Begin spliting data ...")
    datas_list1 = split_data(data_paths[0])
    datas_list2 = split_data(data_paths[1])
    print("Data splited.")

    camp_v1, camp_v2 = get_camp_v(datas_list1, datas_list2)
    print("Camp V1: " + str(camp_v1))
    print("Camp V2: " + str(camp_v2))

    # V1(datas_list1, datas_list2, camp_v1, camp_v2)
    # V2(datas_list1, datas_list2, camp_v1, camp_v2)
    # V3(datas_list1, datas_list2, camp_v1, camp_v2)

def get_camp_v(datas_list1, datas_list2):
    all_train_data1 = [item for sublist in datas_list1[0 : n-1] for item in sublist] # n-1 parts for training
    train_dataset1 = Dataset(all_train_data1, id[0])
    camp_v1 = train_dataset1.statistics['ecpc']

    all_train_data2 = [item for sublist in datas_list2[0 : n-1] for item in sublist]
    train_dataset2 = Dataset(all_train_data2, id[1])
    camp_v2 = train_dataset2.statistics['ecpc']
    return camp_v1, camp_v2

def V1(datas_list1, datas_list2, camp_v1, camp_v2):
    #----------------- data -----------------
    train_dataset1 = Dataset(datas_list1[0], id[0]) 
    train_dataset1.shuffle()
    train_dataset2 = Dataset(datas_list2[0], id[1])
    train_dataset2.shuffle()
    weight = None
    #----------------- train -----------------
    print("Begin training ...")
    for i in range(1, n):
        print("Turn " + str(i) + " ...")
        test_dataset1 = Dataset(datas_list1[i], id[0])
        test_dataset2 = Dataset(datas_list2[i], id[1])
        ctr_estimator = LrModel(train_dataset1, test_dataset1, train_dataset2, test_dataset2, id[0], id[1], camp_v1, camp_v2, weight = weight)
        ctr_estimator.output_data_info()
        for j in range(0, train_round):
            ctr_estimator.train()
            ctr_estimator.test()
            this_round_log = ctr_estimator.get_last_test_log()
            print("Round " + str(j+1) + "\t" + str(this_round_log['performance1']))
            print("Round " + str(j+1) + "\t" + str(this_round_log['performance2']))
        
        weight = ctr_estimator.get_best_test_log()['weight']
        train_dataset1 = ctr_estimator.winning_dataset1
        train_dataset1.shuffle()
        train_dataset2 = ctr_estimator.winning_dataset2
        train_dataset2.shuffle()
        #----------------- output -----------------
        print("Begin log output in this turn ...")
        log_file = "[" + str(i) + "]" + str(id[0]) + "_" + str(id[1]) + "V1.txt"
        os.makedirs(output_path, exist_ok=True)
        log_output_path = output_path + log_file
        ctr_estimator.output_log(log_output_path)

def V2(datas_list1, datas_list2, camp_v1, camp_v2):
    #----------------- data -----------------
    print("Begin loading data ...")
    all_train_data1 = [item for sublist in datas_list1[0 : n-1] for item in sublist] # n-1 parts for training
    train_dataset1 = Dataset(all_train_data1, id[0])
    train_dataset1.shuffle()
    test_dataset1 = Dataset(datas_list1[n-1], id[0])

    all_train_data2 = [item for sublist in datas_list2[0 : n-1] for item in sublist]
    train_dataset2 = Dataset(all_train_data2, id[1])
    train_dataset2.shuffle()
    test_dataset2 = Dataset(datas_list2[n-1], id[1])

    train_dataset1.output_statistics()
    test_dataset1.output_statistics()
    train_dataset2.output_statistics()
    test_dataset2.output_statistics()
    print("Data loaded.")
    #----------------- train -----------------    
    ctr_estimator = LrModel(train_dataset1, test_dataset1, train_dataset2, test_dataset2, id[0], id[1], camp_v1, camp_v2)
    ctr_estimator.output_info()
    ctr_estimator.output_data_info()

    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        print("Round " + str(i+1) + "\t" + str(this_round_log['performance1']))
        print("Round " + str(i+1) + "\t" + str(this_round_log['performance2']))
    print("Train done.")
    #----------------- output -----------------
    log_file = str(id[0]) + "_" + str(id[1]) + "V2.txt"
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

def V3(datas_list1, datas_list2, camp_v1, camp_v2):
    #----------------- data -----------------
    print("Begin loading data ...")
    train_dataset1 = Dataset(datas_list1[0], id[0])
    train_dataset1.shuffle()
    test_dataset1 = Dataset(datas_list1[n-1], id[0])

    train_dataset2 = Dataset(datas_list2[0], id[1])
    train_dataset2.shuffle()
    test_dataset2 = Dataset(datas_list2[n-1], id[1])

    train_dataset1.output_statistics()
    test_dataset1.output_statistics()
    train_dataset2.output_statistics()
    test_dataset2.output_statistics()
    print("Data loaded.")
    #----------------- train -----------------    
    ctr_estimator = LrModel(train_dataset1, test_dataset1, train_dataset2, test_dataset2, id[0], id[1], camp_v1, camp_v2)
    ctr_estimator.output_info()
    ctr_estimator.output_data_info()

    print("Begin training ...")
    for i in range(0, train_round):
        ctr_estimator.train()
        ctr_estimator.test()
        this_round_log = ctr_estimator.get_last_test_log()
        print("Round " + str(i+1) + "\t" + str(this_round_log['performance1']))
        print("Round " + str(i+1) + "\t" + str(this_round_log['performance2']))
    print("Train done.")
    #----------------- output -----------------
    log_file = str(id[0]) + "_" + str(id[1]) + "V3.txt"
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