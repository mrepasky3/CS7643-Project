import torch
from torch import nn
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
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

parser.add_argument('--experiment_type', type=str, choices=["simple","else"])
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
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			data_source : str
				choices are Toy, MNIST, or OMNI
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
					LSTM_optim.zero_grad()
					loss = LSTM_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					LSTM_optim.step()
				elif net == "DS":
					deepsets_optim.zero_grad()
					loss = deepsets_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					deepsets_optim.step()
				elif net == "ATT":
					attention_optim.zero_grad()
					loss = attention_network.compute_loss(bd[i],bm[i],bo[i])
					loss.backward()
					attention_optim.step()

				losses.append(loss.item())
			print('Epoch %d: loss = %1.3e'%(epoch+1,np.mean(losses)))
			elosses.append(np.mean(losses))

		return elosses


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
			LSTM_network.eval()
			preds = LSTM_network(test_data, test_masks)
			MSE = ((pred-test_outs)**2).mean()
			LSTM_network.train()
		elif net == "DS":
			deepsets_network.eval()
			preds = deepsets_network(test_data, test_masks)
			MSE = ((pred-test_outs)**2).mean()
			deepsets_network.train()
		elif net == "ATT":
			attention_network.eval()
			preds = attention_network(test_data, test_masks)
			MSE = ((pred-test_outs)**2).mean()
			attention_network.train()
		return MSE


	def simple_training(self, task, fixed_sizes, save=True):
		"""
		Trains the networks on the same dataset to perform a task,
		then visualize on various fixed-sized sets.

		Parameters
		----------
			task : str
				indicator for the training task
			fixed_sizes : list of int
				fixed set sizes to evaluate upon
			save : bool
				whether or not to save the results

		Returns
		-------
			elosses: list of float
				values of loss after each epoch
		"""

		if save and not os.path.exists("simple_results"):
			os.path.mkdir("simple_results")

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

		data, masks, outs = utils.process_dataset(train_sets, train_labels, max_size=self.dataset.upper_bound-1, source=self.data_source)
		data.to(self.device)
		masks.to(self.device)
		outs.to(self.device)

		test_data, test_masks, test_outs = utils.process_dataset(test_sets, test_labels, max_size=self.dataset.upper_bound-1, source=self.data_source)
		test_data.to(self.device)
		test_masks.to(self.device)
		test_outs.to(self.device)

		LSTM_eloss = self.train_net(data, masks, outs, net="LSTM")
		LSTM_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="LSTM")

		deepsets_eloss = self.train_net(data, masks, outs, net="DS")
		deepsets_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="DS")

		attention_eloss = self.train_net(data, masks, outs, net="ATT")
		attention_test_MSE = self.report_MSE(test_data, test_masks, test_outs, net="ATT")

		if save:
			plt.figure(figsize=(6,5))
			plt.semilogy(LSTM_eloss,'o-', label="LSTM (Test MSE {:.2f})".format(LSTM_test_MSE))
			plt.semilogy(deepsets_eloss,'o-', label="Deep Sets (Test MSE {:.2f})".format(deepsets_test_MSE))
			plt.semilogy(attention_eloss,'o-', label="Self Attention (Test MSE {:.2f})".format(attention_test_MSE))
			plt.legend(fontsize=14)
			plt.xlabel('Epoch',fontsize=14)
			plt.ylabel('Loss',fontsize=14)
			plt.savefig('simple_results/training_curves.png')
			plt.clf()

		LSTM_MSE_fixed = []
		deepsets_MSE_fixed = []
		attention_MSE_fixed = []
		for size in fixed_sizes:
			if data_source == "Toy":
				fixed_dataset = SetsToy(max_size=size, fixed_size=True)
			elif data_source == "MNIST":
				fixed_dataset = SetsMNIST(max_size=size, fixed_size=True)
			elif data_source == "OMNI":
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

			fixed_data, fixed_masks, fixed_outs = utils.process_dataset(fixed_sets, fixed_labels, max_size=fixed_dataset.upper_bound-1, source=self.data_source)
			fixed_data.to(self.device)
			fixed_masks.to(self.device)
			fixed_outs.to(self.device)

			LSTM_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="LSTM"))
			deepsets_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="DS"))
			attention_MSE_fixed.append(self.report_MSE(fixed_data, fixed_masks, fixed_outs, net="ATT"))

		if save:
			plt.figure(figsize=(6,5))
			plt.plot(fixed_sizes, LSTM_MSE_fixed)
			plt.scatter(fixed_sizes, LSTM_MSE_fixed, marker='x', label="LSTM")
			plt.plot(fixed_sizes, deepsets_MSE_fixed)
			plt.scatter(fixed_sizes, deepsets_MSE_fixed, marker='x', label="Deep Sets")
			plt.plot(fixed_sizes, attention_MSE_fixed)
			plt.scatter(fixed_sizes, attention_MSE_fixed, marker='x', label="Self Attention")
			plt.legend(fontsize=14)
			plt.xlabel('Set Size',fontsize=14)
			plt.ylabel('MSE',fontsize=14)
			plt.savefig('simple_results/fixed_size_curves.png')
			plt.clf()

			MSE_results = np.concatenate([np.array(LSTM_MSE_fixed).reshape(-1,1),
				np.array(deepsets_MSE_fixed).reshape(-1,1),
				np.array(attention_MSE_fixed).reshape(-1,1)], axis=1)
			np.save("simple_results/MSE_results.npy", MSE_results)


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
			size_list = np.arange(21,60)
		else:
			size_list = np.arange(3,40)
		if args.suppress_save:
			exp.simple_training(args.task, fixed_sizes=size_list, save=False)
		else:
			exp.simple_training(args.task, fixed_sizes=size_list)

	print('COMPLETED WITHOUT MAJOR ERROR')
