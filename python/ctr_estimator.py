
import copy
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math

from bid_strategy import TruthfulBid
from dataset import Dataset

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
# member variables:
# train_datas: train_datas[i] = [y, z, x1, x2, ...], combined from train_dataset1 and train_dataset2
# test_dataset1, test_dataset2: class Dataset
# winning_dataset1, winning_dataset2: class Dataset, after test become the winning dataset
class LrModel:
	def __init__(self, train_dataset1, test_dataset1, train_dataset2, test_dataset2, id1, id2, camp_v1, camp_v2, weight=None):
		random.seed(10)

		self.id1 = id1
		self.id2 = id2

		self.test_dataset1 = test_dataset1
		self.test_dataset2 = test_dataset2
		self.camp_v1 = camp_v1
		self.camp_v2 = camp_v2

		self.have_budget = have_budget
		self.budget_prop = budget_prop
		self.budget = int(self.test_dataset1.statistics['cost_sum'] + self.test_dataset2.statistics['cost_sum'] / self.budget_prop)

		self.lr_alpha = lr_alpha
		self.lr_lambda = lr_lambda
		if weight is not None:
			self.weight = weight
		else:
			self.weight = {}
		self.best_weight = {}
		self.test_log = []

		self.bid_strategy1 = TruthfulBid(self.camp_v1)
		self.bid_strategy2 = TruthfulBid(self.camp_v2)

		self.train_datas = []
		for data in train_dataset1.datas:
			self.train_datas.append(data)
		for data in train_dataset2.datas:
			self.train_datas.append(data)
		random.shuffle(self.train_datas)

	# pctr = sigmoid(w * x)
	# loss function is Cross Entropy + L2 regularization (1/2 * lambda * w^2)
	# update: w = w - alpha * ((pctr - y) * x + lambda * w)
	#			= w * (1 - alpha * lambda) - alpha * (pctr - y) * x
	def train(self):
		for data in self.train_datas:
			y = data[0]
			feature = data[2:len(data)]

			ctr = estimate_ctr(self.weight, feature, train_flag=True)
			for idx in feature: # update
				self.weight[idx] = self.weight[idx] * (1 - self.lr_alpha * self.lr_lambda) - self.lr_alpha * (ctr - y)

	def test(self):
		parameters = {'weight':self.weight}
		performance1, self.winning_dataset1 = self.calc_performance(self.test_dataset1, parameters, 1)
		performance2, self.winning_dataset2 = self.calc_performance(self.test_dataset2, parameters, 2)

		# record performance
		log = {'weight':copy.deepcopy(self.weight), 'performance1':copy.deepcopy(performance1), 'performance2':copy.deepcopy(performance2)}
		self.test_log.append(log)

	def calc_performance(self, dataset, parameters, index): # calculate the performance w.r.t. the given dataset and parameters
		weight = parameters['weight']
		# budget = parameters['budget']
		bid_sum = 0
		cost_sum = 0
		imp_sum = 0
		clk_sum = 0
		revenue_sum = 0
		labels = []
		p_labels = []
		winning_datas = []

		for data in dataset.datas:
			bid_sum += 1
			y = data[0]
			market_price = data[1]
			feature = data[2:len(data)]

			ctr = estimate_ctr(weight, feature, train_flag=False)
			labels.append(y)
			p_labels.append(ctr)
			
			if index == 1:
				bid_price = self.bid_strategy1.bid(ctr)
			else:
				bid_price = self.bid_strategy2.bid(ctr)
			
			if bid_price > market_price:
				winning_datas.append(data)
				cost_sum += market_price
				imp_sum += 1
				clk_sum += y
				if index == 1:
					revenue_sum = int(revenue_sum - market_price + y * self.camp_v1 * 1E3)
				else:
					revenue_sum = int(revenue_sum - market_price + y * self.camp_v2 * 1E3)
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
		id = self.id1 if index == 1 else self.id2
		winning_data = Dataset(winning_datas, id)
		return performance, winning_data

	def judge_stop(self):
		stop = False
		curr_loop = len(self.test_log) - 1 # the latest record id
		if curr_loop >= 2:
			current_r = self.test_log[curr_loop]['performance1']['revenue'] + self.test_log[curr_loop]['performance2']['revenue']
			last_r = self.test_log[curr_loop - 1]['performance1']['revenue'] + self.test_log[curr_loop - 1]['performance2']['revenue']
			last_2_r = self.test_log[curr_loop - 2]['performance1']['revenue'] + self.test_log[curr_loop - 2]['performance2']['revenue']
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
				revenue = log['performance1']['revenue'] + log['performance2']['revenue']
				if revenue > best_revenue:
					best_revenue = revenue
					best_log = log
		return best_log
	
	def get_best_test_log_index(self): # maximize the revenue
		best_i = 0
		if len(self.test_log) == 0:
			print("ERROR: no record in the log.")
		else:
			best_revenue = -1E10
			for i in range(len(self.test_log)):
				log = self.test_log[i]
				revenue = log['performance1']['revenue'] + log['performance2']['revenue']
				if revenue > best_revenue:
					best_revenue = revenue
					best_i = i
		return best_i		


	def output_weight(self, weight, path):
		fo = open(path, 'w')
		for idx in weight:
			fo.write(str(idx) + '\t' + str(weight[idx]) + '\n')
		fo.close()

	def output_log(self, path):
		fo = open(path, 'w')

		fo.write("Max revenue's round: " + str(self.get_best_test_log_index() + 1) + '\n')
		
		headers = ['round', 'bids', 'cpc', 'cpm', 'ctr', 'revenue', 'imps', 'clks', 'auc', 'rmse', 'roi', 'cost']
		header_line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(*headers)
		fo.write(header_line + '\n')

		for i in range(0, len(self.test_log)):
			test_log = self.test_log[i]
			performance1 = test_log['performance1']
			performance2 = test_log['performance2']
			line1 = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
				i+1,
				performance1['bids'], performance1['cpc'], performance1['cpm'], performance1['ctr'], 
				performance1['revenue'], performance1['imps'], performance1['clks'], performance1['auc'], 
				performance1['rmse'], performance1['roi'], performance1['cost']
			)
			line2 = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
				i+1,
				performance2['bids'], performance2['cpc'], performance2['cpm'], performance2['ctr'], 
				performance2['revenue'], performance2['imps'], performance2['clks'], performance2['auc'], 
				performance2['rmse'], performance2['roi'], performance2['cost']
			)
			line3 = "round " + str(i+1) + ", total revenue = " + str(performance1['revenue'] + performance2['revenue'])
			fo.write(line1 + '\n')
			fo.write(line2 + '\n')
			fo.write(line3 + '\n')

		fo.close()

	def output_info(self):
		print("advertiser 1: " + str(self.id1) + "	its campaign v = " + str(self.camp_v1))
		print("advertiser 2: " + str(self.id2) + "	its campaign v = " + str(self.camp_v2))

		print("lr_alpha = " + str(self.lr_alpha))
		print("lr_lambda = " + str(self.lr_lambda))
		print("have_budget = " + str(self.have_budget))
		if self.have_budget:
			print("budget_prop = " + str(self.budget_prop))
			print("budget = " + str(self.budget))
	
	def output_data_info(self):
		print("train_data's size: " + str(len(self.train_datas)))
		print("test_dataset1's size: " + str(self.test_dataset1.statistics['size']))
		print("test_dataset2's size: " + str(self.test_dataset2.statistics['size']))

