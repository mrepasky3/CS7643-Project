import torch
import torchvision
import torchvision.datasets as datasets

class SetsMNIST:
	"""
	A class to load various set classification tasks on MNIST.
	
	...

	Attributes
	----------
	train_data : tensor
		un-modified training data loaded from MNIST
	train_labels : tensor
		un-modified training labels loaded from MNIST
	test_data : tensor
		un-modified testing data loaded from MNIST
	test_labels : tensor
		un-modified testing labels loaded from MNIST
	train_data_shuffled : tensor
		randomly shuffled training data
	train_labels_shuffled : tensor
		randomly shuffled training labels
	set_bounds : tensor
		random-size bounds on the indices for training sets
	test_data_shuffled : tensor
		randomly shuffled testing data
	test_labels_shuffled : tensor
		randomly shuffled testing labels
	test_set_bounds : tensor
		random-size bounds on the indices for testing sets

	Methods
	-------
	sum_task():
		Returns training and testing data and labels for sum of digits task.
	max_task():
		Returns training and testing data and labels for max of digits task.
	range_task():
		Returns training and testing data and labels for (max - min) of digits task.
	"""
	def __init__(self,seed=None):
		"""
		Loads the data from MNIST, randomly shuffles, and generates set boundaries.

		Parameters
		----------
			seed : int
				random seed for reproducibility
		"""
		if seed is not None:
			torch.manual_seed(seed)

		mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
		mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=None)

		self.train_data, self.train_labels = mnist_train.train_data, mnist_train.train_labels
		self.test_data, self.test_labels = mnist_test.data, mnist_test.targets

		# randomly shuffle the data
		shuffled_idx = torch.randperm(self.train_data.shape[0])
		self.train_data_shuffled = self.train_data[shuffled_idx]
		self.train_labels_shuffled = self.train_labels[shuffled_idx]

		# randomly determine the index range for each set
		random_idx = torch.randint(2,10,(self.train_data_shuffled.shape[0]//2,))
		set_idx = torch.cumsum(random_idx,dim=0)
		set_idx = set_idx[set_idx < self.train_data_shuffled.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.train_data_shuffled.shape[0] - set_idx[-1]) > 1:
		    self.set_bounds = torch.zeros((set_idx.shape[0]+1,2),dtype=torch.int)
		    self.set_bounds[-1] = torch.tensor([set_idx[-1],self.train_data_shuffled.shape[0]])
		else:
		    self.set_bounds = torch.zeros((set_idx.shape[0],2),dtype=torch.int)
		self.set_bounds[0] = torch.tensor([0,set_idx[0]])
		for i in range(set_idx.shape[0]-1):
		    self.set_bounds[i+1] = torch.tensor([set_idx[i],set_idx[i+1]])

		test_shuffled_idx = torch.randperm(self.test_data.shape[0])
		self.test_data_shuffled = self.test_data[test_shuffled_idx]
		self.test_labels_shuffled = self.test_labels[test_shuffled_idx]

		# randomly determine the index range for each set
		test_random_idx = torch.randint(2,10,(self.test_data_shuffled.shape[0]//2,))
		test_set_idx = torch.cumsum(test_random_idx,dim=0)
		test_set_idx = test_set_idx[test_set_idx < self.test_data_shuffled.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.test_data_shuffled.shape[0] - test_set_idx[-1]) > 1:
		    self.test_set_bounds = torch.zeros((test_set_idx.shape[0]+1,2),dtype=torch.int)
		    self.test_set_bounds[-1] = torch.tensor([test_set_idx[-1],self.test_data_shuffled.shape[0]])
		else:
		    self.test_set_bounds = torch.zeros((test_set_idx.shape[0],2),dtype=torch.int)
		self.test_set_bounds[0] = torch.tensor([0,test_set_idx[0]])
		for i in range(test_set_idx.shape[0]-1):
		    self.test_set_bounds[i+1] = torch.tensor([test_set_idx[i],test_set_idx[i+1]])

	def sum_task(self):
		"""
		Returns training and testing data and labels for sum of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains integer sum of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains integer sum of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			train_labels[i] = (self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]).sum()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			test_labels[i] = (self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]).sum()

		return train_sets, train_labels, test_sets, test_labels

	def max_task(self):
		"""
		Returns training and testing data and labels for max of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains integer max of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains integer max of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			train_labels[i] = (self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]).max()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			test_labels[i] = (self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]).max()

		return train_sets, train_labels, test_sets, test_labels

	def range_task(self):
		"""
		Returns training and testing data and labels for (max - min) of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains integer (max - min) of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains integer (max - min) of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = temp_labels.max() - temp_labels.min()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = temp_labels.max() - temp_labels.min()

		return train_sets, train_labels, test_sets, test_labels

	def range_task(self):
		"""
		Returns training and testing data and labels for (max - min) of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains integer (max - min) of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains integer (max - min) of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = temp_labels.max() - temp_labels.min()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = temp_labels.max() - temp_labels.min()

		return train_sets, train_labels, test_sets, test_labels