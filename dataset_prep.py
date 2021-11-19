import torch
import torchvision
import torchvision.datasets as datasets

class SetsToy:
	"""
	A class to load various set classification tasks on sets of digits.
	
	...

	Attributes
	----------
	train_data : tensor
		60,000 random integers from 0 to 9
	test_data : tensor
		10,000 random integers from 0 to 9
	set_bounds : tensor
		random-size bounds on the indices for training sets
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
	mode_task():
		Returns training and testing data and labels for (lowest-valued) mode of digits task.
	product_task():
		Returns training and testing data and labels for product of two largest digits task.
	"""
	def __init__(self,max_size=None,fixed_size=False,seed=None):
		"""
		Generates tensor of random integers and partitions into sets.

		Parameters
		----------
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			fixed_size: bool
				if True, each set has the same
				number of elements
			seed : int
				random seed for reproducibility
		"""
		if seed is not None:
			torch.manual_seed(seed)

		self.train_data = torch.randint(10,(60000,))
		self.test_data = torch.randint(10,(10000,))

		if max_size:
			self.upper_bound = max_size
		else:
			self.upper_bound = int(torch.randint(5,20,(1,)))

		self.fixed_size = fixed_size
		if self.fixed_size:
			self.lower_bound = self.upper_bound
		else:
			self.lower_bound = 2

		# randomly determine the index range for each set
		if self.fixed_size:
			random_idx = torch.ones((self.train_data.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.train_data.shape[0]//2,))
		set_idx = torch.cumsum(random_idx,dim=0)
		set_idx = set_idx[set_idx < self.train_data.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.train_data.shape[0] - set_idx[-1]) > 1:
		    self.set_bounds = torch.zeros((set_idx.shape[0]+1,2),dtype=torch.int)
		    self.set_bounds[-1] = torch.tensor([set_idx[-1],self.train_data.shape[0]])
		else:
		    self.set_bounds = torch.zeros((set_idx.shape[0],2),dtype=torch.int)
		self.set_bounds[0] = torch.tensor([0,set_idx[0]])
		for i in range(set_idx.shape[0]-1):
		    self.set_bounds[i+1] = torch.tensor([set_idx[i],set_idx[i+1]])

		# randomly determine the index range for each test set
		if self.fixed_size:
			test_random_idx = torch.ones((self.test_data.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			test_random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.test_data.shape[0]//2,))
		test_set_idx = torch.cumsum(test_random_idx,dim=0)
		test_set_idx = test_set_idx[test_set_idx < self.test_data.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.test_data.shape[0] - test_set_idx[-1]) > 1:
		    self.test_set_bounds = torch.zeros((test_set_idx.shape[0]+1,2),dtype=torch.int)
		    self.test_set_bounds[-1] = torch.tensor([test_set_idx[-1],self.test_data.shape[0]])
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
			List of variable length training tensors of shape N where N is set size
		train_labels : tensor
			Contains integer sum of digits in training data
		test_sets : list
			List of variable length testing tensors of shape N where N is set size
		test_labels : tensor
			Contains integer sum of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]])
			train_labels[i] = (self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]]).sum()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			test_labels[i] = (self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]).sum()

		return train_sets, train_labels, test_sets, test_labels

	def max_task(self):
		"""
		Returns training and testing data and labels for max of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape N where N is set size
		train_labels : tensor
			Contains integer max of digits in training data
		test_sets : list
			List of variable length testing tensors of shape N where N is set size
		test_labels : tensor
			Contains integer max of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]])
			train_labels[i] = (self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]]).max()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			test_labels[i] = (self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]).max()

		return train_sets, train_labels, test_sets, test_labels

	def range_task(self):
		"""
		Returns training and testing data and labels for (max - min) of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape N where N is set size
		train_labels : tensor
			Contains integer (max - min) of digits in training data
		test_sets : list
			List of variable length testing tensors of shape N where N is set size
		test_labels : tensor
			Contains integer (max - min) of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = temp_labels.max() - temp_labels.min()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = temp_labels.max() - temp_labels.min()

		return train_sets, train_labels, test_sets, test_labels

	def mode_task(self):
		"""
		Returns training and testing data and labels for (lowest-valued) mode of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape N where N is set size
		train_labels : tensor
			Contains lowest-valued integer mode of digits in training data
		test_sets : list
			List of variable length testing tensors of shape N where N is set size
		test_labels : tensor
			Contains lowest-valued integer mode of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = torch.bincount(temp_labels,minlength=10).argmax()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = torch.bincount(temp_labels,minlength=10).argmax()

		return train_sets, train_labels, test_sets, test_labels

	def product_task(self):
		"""
		Returns training and testing data and labels for product of two largest digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape N where N is set size
		train_labels : tensor
			Contains lowest-valued integer mode of digits in training data
		test_sets : list
			List of variable length testing tensors of shape N where N is set size
		test_labels : tensor
			Contains lowest-valued integer mode of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = temp_labels.sort().values[-2:].prod()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = temp_labels.sort().values[-2:].prod()

		return train_sets, train_labels, test_sets, test_labels


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
	mode_task():
		Returns training and testing data and labels for (lowest-valued) mode of digits task.
	product_task():
		Returns training and testing data and labels for product of two largest digits task.
	"""
	def __init__(self,max_size=None,fixed_size=False,seed=None):
		"""
		Loads the data from MNIST, randomly shuffles, and generates set boundaries.

		Parameters
		----------
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			fixed_size: bool
				if True, each set has the same
				number of elements
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

		if max_size:
			self.upper_bound = max_size
		else:
			self.upper_bound = int(torch.randint(5,20,(1,)))

		self.fixed_size = fixed_size
		if self.fixed_size:
			self.lower_bound = self.upper_bound
		else:
			self.lower_bound = 2

		# randomly determine the index range for each set
		if self.fixed_size:
			random_idx = torch.ones((self.train_data_shuffled.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.train_data_shuffled.shape[0]//2,))
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

		# randomly determine the index range for each test set
		if self.fixed_size:
			test_random_idx = torch.ones((self.test_data_shuffled.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			test_random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.test_data_shuffled.shape[0]//2,))
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

	def mode_task(self):
		"""
		Returns training and testing data and labels for (lowest-valued) mode of digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains lowest-valued integer mode of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains lowest-valued integer mode of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = torch.bincount(temp_labels,minlength=10).argmax()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = torch.bincount(temp_labels,minlength=10).argmax()

		return train_sets, train_labels, test_sets, test_labels

	def product_task(self):
		"""
		Returns training and testing data and labels for product of two largest digits task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_labels : tensor
			Contains lowest-valued integer mode of digits in training data
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_labels : tensor
			Contains lowest-valued integer mode of digits in testing data
		"""
		train_sets = []
		train_labels = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]])
			temp_labels = self.train_labels_shuffled[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_labels[i] = temp_labels.sort().values[-2:].prod()

		test_sets = []
		test_labels = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]])
			temp_labels = self.test_labels_shuffled[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_labels[i] = temp_labels.sort().values[-2:].prod()

		return train_sets, train_labels, test_sets, test_labels


class SetsOmniglot:
	"""
	A class to load unique character set classification task on Omniglot.
	
	...

	Attributes
	----------
	train_data : tensor
		training data loaded from Omniglot
	train_labels : tensor
		training labels loaded from Omniglot
	test_data : tensor
		testing data loaded from Omniglot
	test_labels : tensor
		testing labels loaded from Omniglot
	set_bounds : tensor
		random-size bounds on the indices for training sets
	test_set_bounds : tensor
		random-size bounds on the indices for testing sets

	Methods
	-------
	unique_task():
		Returns training and testing data and labels for unique symbols task.
	"""
	def __init__(self,max_size=None,fixed_size=False,seed=None):
		"""
		Loads the data from Omniglot, randomly shuffles, and generates set boundaries.

		Parameters
		----------
			max_size : int
				maximum size of a set, if None default
				to random upper bound on set size
			fixed_size: bool
				if True, each set has the same
				number of elements
			seed : int
				random seed for reproducibility
		"""
		if seed is not None:
			torch.manual_seed(seed)

		omni_train = datasets.Omniglot(root='./data', background=True, download=True, transform=torchvision.transforms.ToTensor())
		omni_test = datasets.Omniglot(root='./data', background=False, download=True, transform=torchvision.transforms.ToTensor())

		data_loader = torch.utils.data.DataLoader(omni_train,
	                                          batch_size=len(omni_train),
	                                          shuffle=True)
		self.train_data, self.train_labels = next(iter(data_loader))

		data_loader = torch.utils.data.DataLoader(omni_test,
	                                          batch_size=len(omni_test),
	                                          shuffle=True)
		self.test_data, self.test_labels = next(iter(data_loader))

		if max_size:
			self.upper_bound = max_size
		else:
			self.upper_bound = int(torch.randint(40,60,(1,)))

		self.fixed_size = fixed_size
		if self.fixed_size:
			self.lower_bound = self.upper_bound
		else:
			self.lower_bound = 20

		# randomly determine the index range for each set
		if self.fixed_size:
			random_idx = torch.ones((self.train_data.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.train_data.shape[0]//20,))
		set_idx = torch.cumsum(random_idx,dim=0)
		set_idx = set_idx[set_idx < self.train_data.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.train_data.shape[0] - set_idx[-1]) > 1:
		    self.set_bounds = torch.zeros((set_idx.shape[0]+1,2),dtype=torch.int)
		    self.set_bounds[-1] = torch.tensor([set_idx[-1],self.train_data.shape[0]])
		else:
		    self.set_bounds = torch.zeros((set_idx.shape[0],2),dtype=torch.int)
		self.set_bounds[0] = torch.tensor([0,set_idx[0]])
		for i in range(set_idx.shape[0]-1):
		    self.set_bounds[i+1] = torch.tensor([set_idx[i],set_idx[i+1]])

		# randomly determine the index range for each test set
		if self.fixed_size:
			test_random_idx = torch.ones((self.test_data.shape[0],),dtype=torch.int) * self.upper_bound
		else:
			test_random_idx = torch.randint(self.lower_bound,self.upper_bound,(self.test_data.shape[0]//20,))
		test_set_idx = torch.cumsum(test_random_idx,dim=0)
		test_set_idx = test_set_idx[test_set_idx < self.test_data.shape[0]]

		# keep all sets with multiple elements, store as tensor where each element is [lower,upper]
		if (self.test_data.shape[0] - test_set_idx[-1]) > 1:
		    self.test_set_bounds = torch.zeros((test_set_idx.shape[0]+1,2),dtype=torch.int)
		    self.test_set_bounds[-1] = torch.tensor([test_set_idx[-1],self.test_data.shape[0]])
		else:
		    self.test_set_bounds = torch.zeros((test_set_idx.shape[0],2),dtype=torch.int)
		self.test_set_bounds[0] = torch.tensor([0,test_set_idx[0]])
		for i in range(test_set_idx.shape[0]-1):
		    self.test_set_bounds[i+1] = torch.tensor([test_set_idx[i],test_set_idx[i+1]])

	def unique_task(self):
		"""
		Returns training and testing data and labels for unique symbols task.

		Returns
		-------
		train_sets : list
			List of variable length training tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		train_targets : tensor
			Contains integer number of unique symbols in training sets
		test_sets : list
			List of variable length testing tensors of shape (N,D,D) where N is 
			set size and D is image dimension
		test_targets : tensor
			Contains integer number of unique symbols in testing sets
		"""
		train_sets = []
		train_targets = torch.zeros(self.set_bounds.shape[0],dtype=torch.int)
		for i in range(self.set_bounds.shape[0]):
			train_sets.append(self.train_data[self.set_bounds[i][0]:self.set_bounds[i][1]].squeeze())
			temp_labels = self.train_labels[self.set_bounds[i][0]:self.set_bounds[i][1]]
			train_targets[i] = len(temp_labels.unique())

		test_sets = []
		test_targets = torch.zeros(self.test_set_bounds.shape[0],dtype=torch.int)
		for i in range(self.test_set_bounds.shape[0]):
			test_sets.append(self.test_data[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]].squeeze())
			temp_labels = self.test_labels[self.test_set_bounds[i][0]:self.test_set_bounds[i][1]]
			test_targets[i] = len(temp_labels.unique())

		return train_sets, train_targets, test_sets, test_targets