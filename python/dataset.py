
import os
import copy
import random

# member variables:
# datas: datas[i] = [y, mp, x1, x2, ...]
# ad_id: the ad_id of the dataset
# statistics: size, cost_sum, clk_sum, ecpm, ecpc, ctr, max_price
# ecpc: cost_sum / clk_sum * 1E-3, so true ecpc(camp_v) is 1000 * the value
class Dataset:
    def __init__(self, datas, ad_id):
        self.datas = datas
        self.ad_id = ad_id
        self.init_statistics()
    
    def shuffle(self):
        random.seed(200)
        random.shuffle(self.datas)
    
    def init_statistics(self):
        self.statistics = {'size':0, 'cost_sum':0, 'clk_sum':0, 'ecpm':0, 'ecpc':0, 'ctr':0.0, 'max_price':0, 'positive_feature_num':0}
        size = 0
        cost_sum = 0
        clk_sum = 0
        max_price = -1
        counter = {}

        for data in self.datas:
            y = data[0]
            mp = data[1]
            size += 1
            cost_sum += mp
            max_price = mp if mp > max_price else max_price
            clk_sum += y
            if y == 1:
                features = data[2:]
                for feature_idx in features:
                    counter[feature_idx] = counter.get(feature_idx, 0) + 1

        self.statistics['size'] = size
        self.statistics['cost_sum'] = cost_sum
        self.statistics['clk_sum'] = clk_sum
        self.statistics['ecpm'] = 1.0 * cost_sum / size
        self.statistics['ecpc'] = int(cost_sum / clk_sum * 1E-3)
        self.statistics['ctr'] = 1.0 * clk_sum / size
        self.statistics['max_price'] = max_price
        self.statistics['positive_feature_num'] = len(counter)

    def output_statistics(self):
        print("Here are the statistics:")
        print(self.statistics)