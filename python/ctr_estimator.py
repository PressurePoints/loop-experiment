
import copy
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math
import numpy as np

from bid_strategy import TruthfulBid, OptimalBid
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

mu_range = np.arange(-0.98, 0.98, 0.04)

# logistic regression model, using loss function with Cross Entropy + L2 regularization
# member variables:
# train_datas: train_datas[i] = [y, z, x1, x2, ...], combined from train_dataset1 and train_dataset2
# test_datasets[i] = class dataset, can be 1 or 2, ids, camp_vs and bid_strategys are similar
# winning_datasets[i] = class Dataset, after test become the winning dataset
# test_log[i] = dictionary{weight, performances}, performances = [performance]
# performance = dictionary{bids, cpc, cpm, ctr, revenue, imps, clks, auc, rmse, roi, cost}
# init_performances = [performance]
# init_winning_datasets = [winning_dataset]
# have_init_weight = boolean  , means whether initial weight is not None
# if initial weight is None, then pctr is always 1/2
class LrModel:
	def initial_test(self):
		self.init_performances = []
		self.init_winning_datasets = []	
		parameters = {'weight':self.weight}	
		for idx in range(len(self.test_datasets)):
			performance, winning_dataset = self.calc_performance(self.test_datasets[idx], parameters, idx)
			self.init_performances.append(performance)
			self.init_winning_datasets.append(winning_dataset)

	def __init__(self, train_datasets, test_datasets, ids, camp_vs, weight=None):
		random.seed(10)

		self.ids = ids
		self.camp_vs = camp_vs
		self.test_datasets = test_datasets

		self.have_budget = have_budget
		self.budget_prop = budget_prop
		self.budgets = [int(dataset.statistics['cost_sum'] / self.budget_prop) for dataset in test_datasets]

		self.lr_alpha = lr_alpha
		self.lr_lambda = lr_lambda
		if weight is not None:
			self.weight = weight
			self.have_init_weight = True
		else:
			self.weight = {}
			self.have_init_weight = False
		self.best_weight = {}
		self.test_log = []

		self.bid_strategys = [TruthfulBid(camp_v) for camp_v in camp_vs]

		self.train_datas = []
		for train_dataset in train_datasets:
			for data in train_dataset.datas:
				self.train_datas.append(data)
		random.shuffle(self.train_datas)

		self.initial_test()

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
		performances = []
		self.winning_datasets = []
		for idx in range(len(self.test_datasets)):
			performance, winning_dataset = self.calc_performance(self.test_datasets[idx], parameters, idx)
			performances.append(performance)
			self.winning_datasets.append(winning_dataset)

		# record performance
		log = {'weight':copy.deepcopy(self.weight), 'performances':copy.deepcopy(performances)}
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
			
			bid_price = self.bid_strategys[index].bid(ctr)
			
			if bid_price > market_price:
				winning_datas.append(data)
				cost_sum += market_price
				imp_sum += 1
				clk_sum += y
				revenue_sum = int(revenue_sum - market_price + y * self.camp_vs[index] * 1E3)
			if self.have_budget and cost_sum >= self.budget:
				break
		cpc = 0.0 if clk_sum == 0 else 1.0 * cost_sum / clk_sum * 1E-3
		cpm = 0.0 if imp_sum == 0 else 1.0 * cost_sum / imp_sum
		ctr = 0.0 if imp_sum == 0 else 1.0 * clk_sum / imp_sum
		winrate = 0.0 if bid_sum == 0 else 1.0 * imp_sum / bid_sum
		roi = 0.0 if cost_sum == 0 else 1.0 * (revenue_sum) / cost_sum
		auc = roc_auc_score(labels, p_labels)
		rmse = math.sqrt(mean_squared_error(labels, p_labels))
		performance = {'bids':bid_sum, 'cpc':cpc, 'cpm':cpm, 
						'ctr': ctr, 'winrate': winrate, 'revenue':revenue_sum, 
						'imps':imp_sum, 'clks':clk_sum,
						'auc': auc, 'rmse': rmse,
						'roi': roi, 'cost': cost_sum}
		winning_dataset = Dataset(winning_datas, self.ids[index])
		return performance, winning_dataset

	def judge_stop(self):
		stop = False
		curr_loop = len(self.test_log) - 1 # the latest record id
		if curr_loop >= 2:
			current_r = sum(performance['revenue'] for performance in self.test_log[curr_loop]['performances'])
			last_r = sum(performance['revenue'] for performance in self.test_log[curr_loop - 1]['performances'])
			last_2_r = sum(performance['revenue'] for performance in self.test_log[curr_loop - 2]['performances'])
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
				revenue = sum(performance['revenue'] for performance in log['performances'])
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
				revenue = sum(performance['revenue'] for performance in self.test_log[i]['performances'])
				if revenue > best_revenue:
					best_revenue = revenue
					best_i = i
		return best_i		


	def output_weight(self, weight, path):
		fo = open(path, 'w')
		for idx in weight:
			fo.write(str(idx) + '\t' + str(weight[idx]) + '\n')
		fo.close()

	def output_log(self, path, scores1 = None, scores2 = None, scores3 = None):
		fo = open(path, 'w')

		best_test_log_index = self.get_best_test_log_index()
		fo.write("Max revenue's round: " + str(best_test_log_index + 1) + '\n')
		fo.write("This round's info:\n")

		# output best round's info		
		headers = ['round', 'bids', 'cpc', 'cpm', 'ctr', 'winrate', 'revenue', 'imps', 'clks', 'auc', 'rmse', 'roi', 'cost']
		header_line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<21} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(*headers)
		fo.write(header_line + '\n')
		test_log = self.test_log[best_test_log_index]
		for performance in test_log['performances']:
			line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<21} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
				best_test_log_index+1,
				performance['bids'], performance['cpc'], performance['cpm'], performance['ctr'], performance['winrate'],
				performance['revenue'], performance['imps'], performance['clks'], performance['auc'], 
				performance['rmse'], performance['roi'], performance['cost']
			)
			fo.write(line + '\n')		
		line2 = "round " + str(best_test_log_index+1) + ", total revenue = " + str(sum(performance['revenue'] for performance in test_log['performances']))
		fo.write(line2 + '\n')
		
		fo.write('\n')
		fo.write('\n')
		if scores1 is not None:
			fo.write("Here are train dataset's quality scores:\n")
			line = '{:<5} {:<20} {:<20} {:<20}'.format('id', 'score1', 'score2', 'score3')
			fo.write(line + '\n')
			for i in range(0, len(self.ids)):
				line = '{:<5} {:<20} {:<20} {:<20}'.format(self.ids[i], scores1[i], scores2[i], scores3[i])
				fo.write(line + '\n')
			fo.write('\n')
			fo.write('\n')		
		fo.write("all rounds' info:\n")

		# output all rounds' info
		fo.write(header_line + '\n')
		for i in range(0, len(self.test_log)):
			test_log = self.test_log[i]
			for performance in test_log['performances']:
				line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<21} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
					i+1,
					performance['bids'], performance['cpc'], performance['cpm'], performance['ctr'], performance['winrate'],
					performance['revenue'], performance['imps'], performance['clks'], performance['auc'], 
					performance['rmse'], performance['roi'], performance['cost']
				)
				fo.write(line + '\n')
			line2 = "round " + str(i+1) + ", total revenue = " + str(sum(performance['revenue'] for performance in test_log['performances']))
			fo.write(line2 + '\n')
		fo.close()

	def output_info(self):
		for i in range(0, len(self.ids)):
			print("advertiser " + str(i+1) + ": " + str(self.ids[i]) + "	its campaign v = " + str(self.camp_vs[i]))
		print("lr_alpha = " + str(self.lr_alpha))
		print("lr_lambda = " + str(self.lr_lambda))
		print("have_budget = " + str(self.have_budget))
		if self.have_budget:
			print("budget_prop = " + str(self.budget_prop))
			for i in range(0, len(self.budgets)):
				print("advertiser " + str(i+1) + " 's budget: " + str(self.budgets[i]))
	
	def output_data_info(self):
		print("train_data's size: " + str(len(self.train_datas)))
		for i in range(0, len(self.test_datasets)):
			print("test_dataset " + str(i+1) + "'s size: " + str(self.test_datasets[i].statistics['size']))

class LrModelWithOptimalBidding(LrModel):
	def __init__(self, train_datasets, test_datasets, ids, camp_vs, weight=None):
		LrModel.__init__(self, train_datasets, test_datasets, ids, camp_vs, weight)
		self.bid_strategys = [OptimalBid(camp_v) for camp_v in self.camp_vs]
		self.mu_range = mu_range
	
	def test(self, whether_search_mu = True):
		parameters = {'weight':self.weight}
		performances = []
		self.winning_datasets = []
		mus = []

		if whether_search_mu:
			# different advertiser use different bidding strategy, that is different mu
			for idx in range(len(self.test_datasets)):
				# print(str(idx+1) + "th advertiser")
				optimal_revenue = -1E10
				optimal_performance = None
				optimal_winning_dataset = None
				optimal_mu = 0.0

				# search for mu
				for mu in self.mu_range:
					# print("mu = " + str(mu))
					self.bid_strategys[idx].set_mu(mu)
					performance, winning_dataset = self.calc_performance(self.test_datasets[idx], parameters, idx)
					if performance['revenue'] > optimal_revenue:
						optimal_revenue = performance['revenue']
						optimal_performance = performance
						optimal_winning_dataset = winning_dataset
						optimal_mu = mu
				
				performances.append(optimal_performance)
				self.winning_datasets.append(optimal_winning_dataset)
				mus.append(optimal_mu)
		else:
			for idx in range(len(self.test_datasets)):
				performance, winning_dataset = self.calc_performance(self.test_datasets[idx], parameters, idx)
				performances.append(performance)
				self.winning_datasets.append(winning_dataset)
				mus.append(-100)

		# record performance
		log = {'weight':copy.deepcopy(self.weight), 'performances':copy.deepcopy(performances), 'mus':copy.deepcopy(mus)}
		self.test_log.append(log)

	def output_log(self, path):
		fo = open(path, 'w')

		fo.write("Max revenue's round: " + str(self.get_best_test_log_index() + 1) + '\n')
		
		headers = ['round', 'bids', 'cpc', 'cpm', 'ctr', 'revenue', 'imps', 'clks', 'auc', 'rmse', 'roi', 'cost']
		header_line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(*headers)
		fo.write(header_line + '\n')

		for i in range(0, len(self.test_log)):
			test_log = self.test_log[i]
			for performance in test_log['performances']:
				line = '{:<5} {:<8} {:<20} {:<20} {:<23} {:<8} {:<8} {:<8} {:<20} {:<20} {:<20} {:<8}'.format(
					i+1,
					performance['bids'], performance['cpc'], performance['cpm'], performance['ctr'], 
					performance['revenue'], performance['imps'], performance['clks'], performance['auc'], 
					performance['rmse'], performance['roi'], performance['cost']
				)
				fo.write(line + '\n')
			line2 = "round " + str(i+1) + ", total revenue = " + str(sum(performance['revenue'] for performance in test_log['performances']))
			line3 = "	optimal mu = "
			for mu in test_log['mus']:
				line3 += str(mu) + ', '
			line3 = line3[:-2]
			fo.write(line2 + line3 + '\n')
		fo.close()
