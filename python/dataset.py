#!/d/anaconda3/envs/python3/python.exe

import os
import copy
import random


# member variables:
# file_path: the path of the dataset file
# statistics: size, cost_sum, clk_sum, ecpm, ecpc, ctr, max_price
# dataset: yzx format dataset   [y, z, x1, x2, ...]
class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.init()
        self.init_index()

    def shuffle(self):
        random.seed(200)
        random.shuffle(self.dataset)

    def init(self):
        print("Begin loading dataset: " + self.file_path)
        self.statistics = {'size':0, 'cost_sum':0, 'clk_sum':0, 'ecpm':0, 'ecpc':0, 'ctr':0.0, 'max_price':0}
        self.dataset = []
        if not os.path.isfile(self.file_path):
            print("ERROR: file not exist. " + self.file_path)
            exit(-1)
        size = 0
        cost_sum = 0
        clk_sum = 0
        max_price = -1
        fi = open(self.file_path, 'r')
        for line in fi:
            li = []
            for d in line.replace(':1','').split():
                li.append(int(d))
            self.dataset.append(li)
            y = li[0]
            mp = li[1]
            size += 1
            cost_sum += mp
            max_price = mp if mp > max_price else max_price
            clk_sum += y
        fi.close()
        self.statistics['size'] = size
        self.statistics['cost_sum'] = cost_sum
        self.statistics['clk_sum'] = clk_sum
        self.statistics['ecpm'] = 1.0 * cost_sum / size
        self.statistics['ecpc'] = int(cost_sum / clk_sum * 1E-3)
        self.statistics['ctr'] = 1.0 * clk_sum / size
        self.statistics['max_price'] = max_price
        print("Loaded. Here are the statistics:")
        print(self.get_statistics())

    def get_statistics(self):
        return self.statistics

    def get_size(self):
        return self.statistics['size']

    def get_max_price(self):
        return self.statistics['max_price']

    def get_dataset(self):
        return self.dataset


    def init_index(self):
        self.iter_index = 0

    def get_next_data(self): # get the next data in the dataset
        if self.iter_index >= self.get_size():
            self.iter_index = 0
        data = self.dataset[self.iter_index]
        self.iter_index = self.iter_index + 1
        return data

    def reached_tail(self): # judge whether the last data have been reached
        flag = (self.iter_index >= self.get_size())
        return flag

# how to iterate the dataset
# dataset = Dataset(file_path)
#
# dataset.init_index()
# while not dataset.reached_tail():
# 	data = dataset.get_next_data()

