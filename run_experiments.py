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

class Experimenter:
	"""
	A class to load unique character set classification task on Omniglot.
	
	...

	Attributes
	----------
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
	unique_task():
		Returns training and testing data and labels for unique symbols task.
	"""
	def __init__(self,data_source,max_size=None,fixed_size=False,lr=1e-3,
		weight_decay=1e-6,batch_size=200,epochs=10,seed=None):
		"""
		Loads the data from Omniglot, randomly shuffles, and generates set boundaries.

		Parameters
		----------
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			data_source : str
				choices are Toy, MNIST, or Omniglot
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
		if seed is not None:
			torch.manual_seed(seed)

		self.batch_size = batch_size
		self.epochs = epochs

		# Load data
		if data_source == "Toy":
			self.dataset = SetsToy(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,1,1])
		elif data_source == "MNIST":
			self.dataset = SetsMNIST(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,28,28])
		elif data_source == "Omniglot":
			self.dataset = SetsOmniglot(max_size=max_size, fixed_size=fixed_size)
			entity_size = torch.tensor([1,105,105])
		
		self.LSTM_network = ImageLSTMNetwork(entity_size=entity_size)
		self.LSTM_optim = torch.optim.Adam(LSTM_network.parameters(), lr=lr, weight_decay=weight_decay)

		self.deepsets_network = ImageDeepSetNetwork(entity_size=entity_size)
		self.deepsets_optim = torch.optim.Adam(deepsets_network.parameters(), lr=lr, weight_decay=weight_decay)

		self.attention_network = ImageAttentionNetwork(entity_size=entity_size)
		self.attention_optim = torch.optim.Adam(attention_network.parameters(), lr=lr, weight_decay=weight_decay)

	def 