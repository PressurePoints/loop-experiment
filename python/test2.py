#!/d/anaconda3/envs/python3/python.exe

from dataset import Dataset
from ctr_estimator import CTREstimator, CTREstimatorFor2

import sys
import os

train_round = 20
id1 = 2259
id2 = 2261

# 1458 3386
# 2259 2261
def main():
    data_folder = "../../make-ipinyou-data/"
    train_path1 = data_folder + str(id1) + "/train.yzx.txt"
    test_path1 = data_folder + str(id1) + "/test.yzx.txt"
    train_path2 = data_folder + str(id2) + "/train.yzx.txt"
    test_path2 = data_folder + str(id2) + "/test.yzx.txt"

    train_data1 = Dataset(train_path1)
    train_data1.shuffle()
    test_data1 = Dataset(test_path1)
    train_data2 = Dataset(train_path2)
    train_data2.shuffle()
    test_data2 = Dataset(test_path2)
    print('Load done.')

    ctr_estimator = CTREstimatorFor2(train_data1, test_data1, train_data2, test_data2, id1, id2)
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


    log_file = str(id1) + "_" + str(id2) + "_" + str(ctr_estimator.lr_alpha) + "_" + str(ctr_estimator.budget_prop) + ".txt"
    os.makedirs("../output", exist_ok=True)
    log_output_path = "../output/" + log_file
    print("Begin log output ...")
    ctr_estimator.output_log(log_output_path)
    print("Log output done.")

    print("Begin weight output ...")
    weight_path = str(id1) + "_" + str(id2) + "_" + "best_weight" \
                + "_" + str(ctr_estimator.lr_alpha) + "_" + str(ctr_estimator.budget_prop) \
                + ".weight"
    best_test_log = ctr_estimator.get_best_test_log()
    ctr_estimator.output_weight(best_test_log['weight'], "../output/" + weight_path)
    print("Weight output done.")


if __name__ == '__main__':
    main()