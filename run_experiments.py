import torch
from torch import nn
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import arrow
from tqdm.notebook import tqdm
from dataset_prep import SetsToy, SetsMNIST, SetsOmniglot
from AttentionNetwork import ImageAttentionNetwork
from DeepSetNetwork import ImageDeepSetNetwork
from LSTMNetwork import ImageLSTMNetwork
import utils
import argparse
import os

parser = argparse.ArgumentParser()

# initializer arguments
parser.add_argument('--data_source', type=str, choices=["Toy","MNIST","OMNI"])
parser.add_argument('--max_size', type=int, default=0)
parser.add_argument('--fixed_size', action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--experiment_type', type=str, choices=["simple","times","batch_times"])
parser.add_argument('--suppress_save', action="store_true")

# simple training arguments
parser.add_argument('--task', type=str, default='sum', choices=['sum','max','range','mode','product','unique'])

class Experimenter:
	"""
	A class to run experiments.
	
	...

	Attributes
	----------
	data_source : str
		indicator for chosen dataset
	batch_size : int
		batch size for training
	epochs : int
		number of training epochs
	LSTM_network : torch.nn.Module
		LSTM network to be trained
	LSTM_optim : torch.optim.Adam
		optimizer for the LSTM
	deepsets_network : torch.nn.Module
		deep sets network to be trained
	deepsets_optim : torch.optim.Adam
		optimizer for the deep sets network
	attention_network : torch.nn.Module
		self-attention network to be trained
	attention_optim : torch.optim.Adam
		optimizer for the self-attention network

	Methods
	-------
	simple_training():
		Trains the networks on the same dataset to perform a task,
		then visualize on various fixed-sized sets.
	"""
	def __init__(self,data_source,max_size=None,fixed_size=False,lr=1e-3,
		weight_decay=1e-6,batch_size=200,epochs=10,seed=None):
		"""
		Loads chosen datasets and initializes the networks and optimizers.

		Parameters
		----------
			data_source : str
				choices are Toy, MNIST, or OMNI
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			fixed_size : bool
				if True, each set has the same
				number of elements
			lr : float
				learning rate of neural networks
			dataset : dataset object
				dataset object used to collect
				data for the learning task
			weight_decay : float
				weight decay for neural networks
			batch_size : int
				batch size for training
			epochs : int
				number of training epochs
			seed : int
				random seed for reproducibility
		"""

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		if seed is not None:
			torch.manual_seed(seed)

		self.data_source = data_source
		self.batch_size = batch_size
		self.epochs = epochs

		# Load data
		if data_source == "Toy":
			self.dataset = SetsToy(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,1,1])
		elif data_source == "MNIST":
			self.dataset = SetsMNIST(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,28,28])
		elif data_source == "OMNI":
			self.dataset = SetsOmniglot(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,105,105])
		
		self.LSTM_network = ImageLSTMNetwork(entity_size=entity_size).to(self.device)
		self.LSTM_optim = torch.optim.Adam(self.LSTM_network.parameters(), lr=lr, weight_decay=weight_decay)

		self.deepsets_network = ImageDeepSetNetwork(entity_size=entity_size).to(self.device)
		self.deepsets_optim = torch.optim.Adam(self.deepsets_network.parameters(), lr=lr, weight_decay=weight_decay)

		self.attention_network = ImageAttentionNetwork(entity_size=entity_size).to(self.device)
		self.attention_optim = torch.optim.Adam(self.attention_network.parameters(), lr=lr, weight_decay=weight_decay)


	def train_net(self, data, masks, outs, net):
		"""
		Choose a neural network to train on given data.

		Parameters
		----------
			data : torch.Tensor
				tensor of training data
			masks : torch.Tensor
				indicators for non-present instances
				in a set
			outs : torch.Tensor
				labels for the training data
			net : str
				choices are LSTM, DS, and ATT

		Returns
		-------
			elosses: list of float
				values of loss after each epoch
		"""

		elosses = []

		for epoch in range(self.epochs):
			bd, bm, bo = utils.batch_data(data,masks,outs,bs=self.batch_size)
			losses = []
			for i in tqdm(range(len(bd))):
				if net == "LSTM":
					self.LSTM_optim.zero_grad()
					loss = self.LSTM_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.LSTM_optim.step()
				elif net == "DS":
					self.deepsets_optim.zero_grad()
					loss = self.deepsets_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.deepsets_optim.step()
				elif net == "ATT":
					self.attention_optim.zero_grad()
					loss = self.attention_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.attention_optim.step()

				losses.append(loss.item())
			print('Epoch %d: loss = %1.3e'%(epoch+1,np.mean(losses)))
			elosses.append(np.mean(losses))

		return elosses


	def report_train_time(self, data, masks, outs, net):
		"""
		Choose a neural network to train on given data
		for 10 epochs and return the average time.

		Parameters
		----------
			data : torch.Tensor
				tensor of training data
			masks : torch.Tensor
				indicators for non-present instances
				in a set
			outs : torch.Tensor
				labels for the training data
			net : str
				choices are LSTM, DS, and ATT

		Returns
		-------
			time: float
				average duration in seconds of training
				for a single epoch
		"""
		times = []
		for _ in range(10):
			start_time = arrow.now()

			bd, bm, bo = utils.batch_data(data,masks,outs,bs=self.batch_size)
			losses = []
			for i in tqdm(range(len(bd))):
				if net == "LSTM":
					self.LSTM_optim.zero_grad()
					loss = self.LSTM_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.LSTM_optim.step()
				elif net == "DS":
					self.deepsets_optim.zero_grad()
					loss = self.deepsets_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.deepsets_optim.step()
				elif net == "ATT":
					self.attention_optim.zero_grad()
					loss = self.attention_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.attention_optim.step()

				losses.append(loss.item())

			times.append((arrow.now() - start_time).total_seconds())
		return np.array(times).mean()


	def report_batch_time(self, data, masks, outs, net):
		"""
		Choose a neural network to train on given data
		for 5 epochs and return the average time of each batch.

		Parameters
		----------
			data : torch.Tensor
				tensor of training data
			masks : torch.Tensor
				indicators for non-present instances
				in a set
			outs : torch.Tensor
				labels for the training data
			net : str
				choices are LSTM, DS, and ATT

		Returns
		-------
			time: float
				average duration in seconds of training
				for a single epoch
		"""
		times = []
		for _ in range(5):

			bd, bm, bo = utils.batch_data(data,masks,outs,bs=self.batch_size)
			losses = []
			for i in tqdm(range(len(bd))):
				start_time = arrow.now()
				if net == "LSTM":
					self.LSTM_optim.zero_grad()
					loss = self.LSTM_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.LSTM_optim.step()
				elif net == "DS":
					self.deepsets_optim.zero_grad()
					loss = self.deepsets_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.deepsets_optim.step()
				elif net == "ATT":
					self.attention_optim.zero_grad()
					loss = self.attention_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					self.attention_optim.step()

				losses.append(loss.item())

				times.append((arrow.now() - start_time).total_seconds())
		return np.array(times).mean()


	def report_MSE(self, test_data, test_masks, test_outs, net):
		"""
		Choose a neural network to evaluate on testing data.

		Parameters
		----------
			test_data : torch.Tensor
				tensor of training data
			test_masks : torch.Tensor
				indicators for non-present instances
				in a set
			test_outs : torch.Tensor
				labels for the training data
			net : str
				choices are LSTM, DS, and ATT

		Returns
		-------
			MSE: float
				MSE of net on testing set
		"""
		if net == "LSTM":
			self.LSTM_network.eval()
			preds = self.LSTM_network(test_data, test_masks)
			MSE = ((preds.detach()-test_outs)**2).mean()
			self.LSTM_network.train()
		elif net == "DS":
			self.deepsets_network.eval()
			preds = self.deepsets_network(test_data, test_masks)
			MSE = ((preds.detach()-test_outs)**2).mean()
			self.deepsets_network.train()
		elif net == "ATT":
			self.attention_network.eval()
			preds = self.attention_network(test_data, test_masks)
			MSE = ((preds.detach()-test_outs)**2).mean()
			self.attention_network.train()
		return MSE


	def report_MSE_OMNI(self, test_data, test_masks, test_outs, net):
		"""
		Choose a neural network to evaluate on testing data.

		Parameters
		----------
			test_data : torch.Tensor
				tensor of training data
			test_masks : torch.Tensor
				indicators for non-present instances
				in a set
			test_outs : torch.Tensor
				labels for the training data
			net : str
				choices are LSTM, DS, and ATT

		Returns
		-------
			MSE: float
				MSE of net on testing set
		"""
		bd, bm, bo = utils.batch_data(test_data,test_masks,test_outs,bs=10)
		MSE = []
		for i in tqdm(range(len(bd))):
			if net == "LSTM":
				self.LSTM_network.eval()
				preds = self.LSTM_network(bd[i], bm[i])
				MSE.append(((preds.detach()-bo[i])**2).mean())
				self.LSTM_network.train()
			elif net == "DS":
				self.deepsets_network.eval()
				preds = self.deepsets_network(bd[i], bm[i])
				MSE.append(((preds.detach()-bo[i])**2).mean())
				self.deepsets_network.train()
			elif net == "ATT":
				self.attention_network.eval()
				preds = self.attention_network(bd[i], bm[i])
				MSE.append(((preds.detach()-bo[i])**2).mean())
				self.attention_network.train()
		return torch.tensor(MSE).mean()


	def simple_training(self, task, fixed_sizes, savepath=None):
		"""
		Trains the networks on the same dataset to perform a task,
		then visualize on various fixed-sized sets.

		Parameters
		----------
			task : str
				indicator for the training task
			fixed_sizes : list of int
				fixed set sizes to evaluate upon
			savepath : bool
				directory to save the results
		"""

		if savepath is not None:
			if not os.path.exists("simple_results"):
				os.mkdir("simple_results")
			savepath = "simple_results/"+savepath
			if not os.path.exists(savepath):
				os.mkdir(savepath)


		if task == 'sum':
			train_sets, train_labels, test_sets, test_labels = self.dataset.sum_task()
		elif task == 'max':
			train_sets, train_labels, test_sets, test_labels = self.dataset.max_task()
		elif task == 'range':
			train_sets, train_labels, test_sets, test_labels = self.dataset.range_task()
		elif task == 'mode':
			train_sets, train_labels, test_sets, test_labels = self.dataset.mode_task()
		elif task == 'product':
			train_sets, train_labels, test_sets, test_labels = self.dataset.product_task()
		elif task == 'unique':
			train_sets, train_labels, test_sets, test_labels = self.dataset.unique_task()

		data, masks, outs = utils.process_dataset(train_sets, train_labels, max_size=self.dataset.upper_bound, source=self.data_source)
		data = data.to(self.device)
		masks = masks.to(self.device)
		outs = outs.to(self.device)

		test_data, test_masks, test_outs = utils.process_dataset(test_sets, test_labels, max_size=self.dataset.upper_bound, source=self.data_source)
		test_data = test_data.to(self.device)
		test_masks = test_masks.to(self.device)
		test_outs = test_outs.to(self.device)
		
		if self.device == "cuda:0":
			del train_sets
			del train_labels
			del test_sets
			del test_labels
			torch.cuda.empty_cache()

		LSTM_eloss = self.train_net(data, masks, outs, net="LSTM")
		if self.data_source == "OMNI":
			LSTM_test_MSE = self.report_MSE_OMNI(test_data, test_masks, test_outs, net="LSTM")
		else:
			LSTM_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="LSTM")

		deepsets_eloss = self.train_net(data, masks, outs, net="DS")
		if self.data_source == "OMNI":
			deepsets_test_MSE = self.report_MSE_OMNI(test_data, test_masks, test_outs, net="DS")
		else:
			deepsets_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="DS")

		attention_eloss = self.train_net(data, masks, outs, net="ATT")
		if self.data_source == "OMNI":
			attention_test_MSE = self.report_MSE_OMNI(test_data, test_masks, test_outs, net="ATT")
		else:
			attention_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="ATT")

		if savepath is not None:
			plt.figure(figsize=(9,7))
			plt.plot(LSTM_eloss)
			plt.scatter(np.arange(len(LSTM_eloss)),LSTM_eloss, marker='x', label="LSTM (Test MSE {:.2f})".format(LSTM_test_MSE))
			plt.plot(deepsets_eloss)
			plt.scatter(np.arange(len(deepsets_eloss)),deepsets_eloss, marker='x', label="Deep Sets (Test MSE {:.2f})".format(deepsets_test_MSE))
			plt.plot(attention_eloss)
			plt.scatter(np.arange(len(attention_eloss)),attention_eloss, marker='x', label="Self Attention (Test MSE {:.2f})".format(attention_test_MSE))
			plt.yscale('log')
			plt.legend(fontsize=14)
			plt.xlabel('Epoch',fontsize=14)
			plt.ylabel('Loss',fontsize=14)
			plt.savefig(savepath+'/training_curves.png')
			plt.clf()

		LSTM_MSE_fixed = []
		deepsets_MSE_fixed = []
		attention_MSE_fixed = []
		for size in fixed_sizes:
			if self.data_source == "Toy":
				fixed_dataset = SetsToy(max_size=size, fixed_size=True)
			elif self.data_source == "MNIST":
				fixed_dataset = SetsMNIST(max_size=size, fixed_size=True)
			elif self.data_source == "OMNI":
				fixed_dataset = SetsOmniglot(max_size=size, fixed_size=True)

			if task == 'sum':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.sum_task()
			elif task == 'max':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.max_task()
			elif task == 'range':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.range_task()
			elif task == 'mode':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.mode_task()
			elif task == 'product':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.product_task()
			elif task == 'unique':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.unique_task()

			fixed_data, fixed_masks, fixed_outs = utils.process_dataset(fixed_sets, fixed_labels, max_size=fixed_dataset.upper_bound, source=self.data_source)
			fixed_data = fixed_data.to(self.device)
			fixed_masks = fixed_masks.to(self.device)
			fixed_outs = fixed_outs.to(self.device)

			if self.data_source == "OMNI":
				LSTM_MSE_fixed.append(self.report_MSE_OMNI(fixed_data, fixed_masks, fixed_outs, net="LSTM"))
				deepsets_MSE_fixed.append(self.report_MSE_OMNI(fixed_data, fixed_masks, fixed_outs, net="DS"))
				attention_MSE_fixed.append(self.report_MSE_OMNI(fixed_data, fixed_masks, fixed_outs, net="ATT"))
			else:
				LSTM_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="LSTM"))
				deepsets_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="DS"))
				attention_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="ATT"))

			if self.device == "cuda:0":
				del fixed_sets
				del fixed_labels
				del fixed_data
				del fixed_masks
				del fixed_outs
				torch.cuda.empty_cache()

		if savepath is not None:
			plt.figure(figsize=(9,7))
			plt.plot(fixed_sizes, torch.tensor(LSTM_MSE_fixed).detach())
			plt.scatter(fixed_sizes, torch.tensor(LSTM_MSE_fixed).detach(), marker='x', label="LSTM")
			plt.plot(fixed_sizes, torch.tensor(deepsets_MSE_fixed).detach())
			plt.scatter(fixed_sizes, torch.tensor(deepsets_MSE_fixed).detach(), marker='x', label="Deep Sets")
			plt.plot(fixed_sizes, torch.tensor(attention_MSE_fixed).detach())
			plt.scatter(fixed_sizes, torch.tensor(attention_MSE_fixed).detach(), marker='x', label="Self Attention")
			plt.legend(fontsize=14)
			plt.xlabel('Set Size',fontsize=14)
			plt.ylabel('MSE',fontsize=14)
			plt.savefig(savepath+'/fixed_size_curves.png')
			plt.clf()

			MSE_results = np.concatenate([fixed_sizes.reshape(-1,1),
				np.array(torch.tensor(LSTM_MSE_fixed).detach()).reshape(-1,1),
				np.array(torch.tensor(deepsets_MSE_fixed).detach()).reshape(-1,1),
				np.array(torch.tensor(attention_MSE_fixed).detach()).reshape(-1,1)], axis=1)
			np.save(savepath+"/MSE_results.npy", MSE_results)


	def time_experiments(self, task, fixed_sizes, savepath=None, time_type='epoch'):
		"""
		Trains the networks various fixed-sized sets for one
		epoch and visualize the duration of one epoch for each
		of the fixed set sizes.

		Parameters
		----------
			task : str
				indicator for the training task
			fixed_sizes : list of int
				fixed set sizes to evaluate upon
			savepath : bool
				directory to save the results
		"""

		if savepath is not None:
			if not os.path.exists("simple_results"):
				os.mkdir("simple_results")
			savepath = "simple_results/"+savepath
			if not os.path.exists(savepath):
				os.mkdir(savepath)

		LSTM_times = []
		deepsets_times = []
		attention_times = []
		for size in fixed_sizes:
			if self.data_source == "Toy":
				fixed_dataset = SetsToy(max_size=size, fixed_size=True)
			elif self.data_source == "MNIST":
				fixed_dataset = SetsMNIST(max_size=size, fixed_size=True)
			elif self.data_source == "OMNI":
				fixed_dataset = SetsOmniglot(max_size=size, fixed_size=True)

			if task == 'sum':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.sum_task()
			elif task == 'max':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.max_task()
			elif task == 'range':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.range_task()
			elif task == 'mode':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.mode_task()
			elif task == 'product':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.product_task()
			elif task == 'unique':
				fixed_sets, fixed_labels, _, _ = fixed_dataset.unique_task()

			fixed_data, fixed_masks, fixed_outs = utils.process_dataset(fixed_sets, fixed_labels, max_size=fixed_dataset.upper_bound, source=self.data_source)
			fixed_data = fixed_data.to(self.device)
			fixed_masks = fixed_masks.to(self.device)
			fixed_outs = fixed_outs.to(self.device)

			if time_type == 'epoch':
				LSTM_times.append(self.report_train_time(fixed_data, fixed_masks, fixed_outs, net="LSTM"))
				deepsets_times.append(self.report_train_time(fixed_data, fixed_masks, fixed_outs, net="DS"))
				attention_times.append(self.report_train_time(fixed_data, fixed_masks, fixed_outs, net="ATT"))
			elif time_type == 'batch':
				LSTM_times.append(self.report_batch_time(fixed_data, fixed_masks, fixed_outs, net="LSTM"))
				deepsets_times.append(self.report_batch_time(fixed_data, fixed_masks, fixed_outs, net="DS"))
				attention_times.append(self.report_batch_time(fixed_data, fixed_masks, fixed_outs, net="ATT"))

			if self.device == "cuda:0":
				del fixed_sets
				del fixed_labels
				del fixed_data
				del fixed_masks
				del fixed_outs
				torch.cuda.empty_cache()

		if savepath is not None:
			plt.figure(figsize=(9,7))
			plt.plot(fixed_sizes, LSTM_times)
			plt.scatter(fixed_sizes, LSTM_times, marker='x', label="LSTM")
			plt.plot(fixed_sizes, deepsets_times)
			plt.scatter(fixed_sizes, deepsets_times, marker='x', label="Deep Sets")
			plt.plot(fixed_sizes, attention_times)
			plt.scatter(fixed_sizes, attention_times, marker='x', label="Self Attention")
			plt.legend(fontsize=14)
			plt.xlabel('Set Size',fontsize=14)
			if time_type == 'epoch':
				plt.ylabel('Train Time',fontsize=14)
				plt.savefig(savepath+'/train_time_curves.png')
			elif time_type == 'epoch':
				plt.ylabel('Batch Train Time',fontsize=14)
				plt.savefig(savepath+'/batch_time_curves.png')
			plt.clf()


if __name__ == "__main__":
	args = parser.parse_args()

	if args.max_size == 0:
		args.max_size = None

	if args.fixed_size:
		fixed_size = True
	else:
		fixed_size = False

	if args.seed == 0:
		args.seed = None

	exp = Experimenter(data_source=args.data_source,max_size=args.max_size,fixed_size=fixed_size,lr=args.lr,
		weight_decay=args.weight_decay,batch_size=args.batch_size,epochs=args.epochs,seed=args.seed)

	if args.experiment_type == 'simple':
		if args.data_source == "OMNI":
			size_list = np.arange(21,40)
		else:
			size_list = np.arange(2,60)
		if args.suppress_save:
			exp.simple_training(args.task, fixed_sizes=size_list)
		else:
			exp.simple_training(args.task, fixed_sizes=size_list, savepath=args.data_source+"_"+args.task)

	elif args.experiment_type == 'times':
		if args.data_source == "OMNI":
			size_list = np.arange(21,40)
		else:
			size_list = np.arange(2,60)
		if args.suppress_save:
			exp.time_experiments(args.task, fixed_sizes=size_list)
		else:
			exp.time_experiments(args.task, fixed_sizes=size_list, savepath=args.data_source+"_"+args.task)

	elif args.experiment_type == 'batch_times':
		if args.data_source == "OMNI":
			size_list = np.arange(21,40)
		else:
			size_list = np.arange(2,60)
		if args.suppress_save:
			exp.time_experiments(args.task, fixed_sizes=size_list, time_type='batch')
		else:
			exp.time_experiments(args.task, fixed_sizes=size_list, savepath=args.data_source+"_"+args.task, time_type='batch')

	print('COMPLETED WITHOUT MAJOR ERROR')
