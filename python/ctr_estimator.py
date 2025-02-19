#!/d/anaconda3/envs/python3/python.exe

import copy
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math

from bid_strategy import BidStrategy

def sigmoid(x):
	value = 0.5
	try:
		value = 1.0 / (1.0 + math.exp(-x))
	except:
		value = 1E-9
	return value

def random_weight():
    init_weight = 0.05
    return (random.random() - 0.5) * init_weight

def estimate_ctr(weight, feature, train_flag = False):
	value = 0.0
	for idx in feature:
		if idx in weight:
			value += weight[idx]
		elif train_flag:
			weight[idx] = random_weight()
	ctr = sigmoid(value)
	return ctr


budget_prop = 1
have_budget = False
lr_alpha = 1E-4
lr_lambda = 1E-4
# logistic regression model
class CTREstimator:
	def __init__(self, train_data, test_data):
		self.train_data = train_data
		self.test_data = test_data
		self.camp_v = self.train_data.get_statistics()['ecpc']
		self.have_budget = have_budget
		self.budget_prop = budget_prop
		self.budget = int(self.test_data.get_statistics()['cost_sum'] / self.budget_prop)
		self.lr_alpha = lr_alpha
		self.lr_lambda = lr_lambda
		self.weight = {}
		self.best_weight = {}
		self.test_log = []

		self.bid_strategy = BidStrategy(self.camp_v)

	def train(self): # train with one traversal of the full train_data
		random.seed(10)
		train_data = self.train_data

		train_data.init_index()
		while not train_data.reached_tail():
			data = train_data.get_next_data()
			y = data[0]
			feature = data[2:len(data)]

			ctr = estimate_ctr(self.weight, feature, train_flag=True)
			
			for idx in feature: # update
				self.weight[idx] = self.weight[idx] * (1 - self.lr_alpha * self.lr_lambda) - self.lr_alpha * (ctr - y)
		# print(self.weight)

	def test(self):
		parameters = {'weight':self.weight}
		performance = self.calc_performance(self.test_data, parameters)

		# record performance
		log = {'weight':copy.deepcopy(self.weight), 'performance':copy.deepcopy(performance)}
		self.test_log.append(log)

	def calc_performance(self, dataset, parameters): # calculate the performance w.r.t. the given dataset and parameters
		weight = parameters['weight']
		# budget = parameters['budget']
		bid_sum = 0
		cost_sum = 0
		imp_sum = 0
		clk_sum = 0
		revenue_sum = 0
		labels = []
		p_labels = []

		dataset.init_index()
		while not dataset.reached_tail(): #TODO no budget set
			bid_sum += 1
			data = dataset.get_next_data()
			y = data[0]
			market_price = data[1]
			feature = data[2:len(data)]

			ctr = estimate_ctr(weight, feature, train_flag=False)
			labels.append(y)
			p_labels.append(ctr)
			
			bid_price = self.bid_strategy.bid(ctr)
			if bid_price > market_price:
				cost_sum += market_price
				imp_sum += 1
				clk_sum += y
				revenue_sum = int(revenue_sum - market_price + y * self.camp_v * 1E3)
			if self.have_budget and cost_sum >= self.budget:
				break
		cpc = 0.0 if clk_sum == 0 else 1.0 * cost_sum / clk_sum * 1E-3
		cpm = 0.0 if imp_sum == 0 else 1.0 * cost_sum / imp_sum
		ctr = 0.0 if imp_sum == 0 else 1.0 * clk_sum / imp_sum
		roi = 0.0 if cost_sum == 0 else 1.0 * (revenue_sum) / cost_sum
		auc = roc_auc_score(labels, p_labels)
		rmse = math.sqrt(mean_squared_error(labels, p_labels))
		performance = {'bids':bid_sum, 'cpc':cpc, 'cpm':cpm, 
						'ctr': ctr, 'revenue':revenue_sum, 
						'imps':imp_sum, 'clks':clk_sum,
						'auc': auc, 'rmse': rmse,
						'roi': roi, 'cost': cost_sum}
		return performance

	def judge_stop(self):
		stop = False
		# step = int(1/config.train_progress_unit)
		step = 1
		curr_loop = len(self.test_log) - 1 # the latest record id
		if curr_loop >= 2*step:
			current_r = self.test_log[curr_loop]['performance']['revenue']
			last_r = self.test_log[curr_loop - step]['performance']['revenue']
			last_2_r = self.test_log[curr_loop - 2*step]['performance']['revenue']
			# print "Curr:last:last_2 = " + `current_r` + ":" + `last_r` + ":" + `last_2_r`
			if current_r < last_r and last_r < last_2_r:
				stop = True
		return stop

	def get_last_test_log(self):
		return self.test_log[len(self.test_log)-1]

	def get_best_test_log(self): # maximize the revenue
		best_log = {}
		if len(self.test_log) == 0:
			print("ERROR: no record in the log.")
		else:
			best_revenue = -1E10
			for log in self.test_log:
				revenue = log['performance']['revenue']
				if revenue > best_revenue:
					best_revenue = revenue
					best_log = log
		return best_log


	def output_weight(self, weight, path):
		fo = open(path, 'w')
		for idx in weight:
			fo.write(str(idx) + '\t' + str(weight[idx]) + '\n')
		fo.close()

	def output_log(self, path):
		fo = open(path, 'w')
		
		headers = ['round', 'bids', 'cpc', 'cpm', 'ctr', 'revenue', 'imps', 'clks', 'auc', 'rmse', 'roi', 'cost']
		header_line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(*headers)
		fo.write(header_line + '\n')

		for i in range(0, len(self.test_log)):
			test_log = self.test_log[i]
			performance = test_log['performance']
			line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
				i+1,
				performance['bids'], performance['cpc'], performance['cpm'], performance['ctr'], 
				performance['revenue'], performance['imps'], performance['clks'], performance['auc'], 
				performance['rmse'], performance['roi'], performance['cost']
			)
			fo.write(line + '\n')

		fo.close()

	def output_info(self):
		print("campaign v = " + str(self.camp_v))
		print("lr_alpha = " + str(self.lr_alpha))
		print("lr_lambda = " + str(self.lr_lambda))
		print("have_budget = " + str(self.have_budget))
		if self.have_budget:
			print("budget_prop = " + str(self.budget_prop))
			print("budget = " + str(self.budget))
