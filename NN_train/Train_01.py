import os

from sklearn import datasets
import torch

from torch import nn
import sys
import random

import os
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from   librosa import amplitude_to_db
from   librosa import stft
import librosa.display 
import numpy as np
import librosa
import torch.nn.functional as F


# sys.path.insert(0, '/home/lofu/Documents/working_dir/missile_speed/Sound_Manipulation')
sys.path.append('/home/lofu/Documents/working_dir/missile_speed/Sound_To_NNinput/')


from Transform import Transform_For_Input

from torch.utils.data import DataLoader

class ConvNet(nn.Module):
	'''
		Simple Convolutional Neural Network
	'''
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		self.fc3 = nn.Linear(10, 1)
		self.Sig = nn.Sigmoid()
		


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, (2, 2))
		# If the size is a square, you can specify with a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.Sig(x)
		return x
  
  
if __name__ == '__main__':
	print("+I+")
	
	# # Set fixed random number seed
	torch.manual_seed(42)
	convnet = ConvNet()  
	batch_size = 10
	num_classes = 2
	learning_rate = 0.001
	num_epochs = 20
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(convnet)
	print(device)
	
	train_data_dir = "/home/lofu/Documents/working_dir/data/Combandnorocket1_3000-003/comb/"
	f=[]
	for (dirpath, dirnames, filenames) in os.walk(train_data_dir):
		for i in filenames:
			if ".wav" in i:
				f.append(i)
	# a = [[1,2],[1,2],[1,2],[1,2],[1,13],[4,3],]
	# for i,(j,k) in enumerate(a):
		# print(i,j,k)

	train_dataset = []
	counter = 0
	for i in f:
		if counter < 2*batch_size:
			counter+=1
			train_dataset.append( [Transform_For_Input(train_data_dir+i) ,1])
		# print(train_dataset.shape)
		
		
	# print(train_dataset[0].shape)
	print(train_dataset[0][0].shape)
	train_data_dir = "/home/lofu/Documents/working_dir/data/Combandnorocket1_3000-003/norocket/"
	f=[]
	for (dirpath, dirnames, filenames) in os.walk(train_data_dir):
		for i in filenames:
			if ".wav" in i:
				f.append(i)
	# a = [[1,2],[1,2],[1,2],[1,2],[1,13],[4,3],]
	# for i,(j,k) in enumerate(a):
		# print(i,j,k)

	
	for i in f:
		if counter < 3*batch_size:
			counter+=1
			train_dataset.append( [Transform_For_Input(train_data_dir+i) ,0])
		# print(train_dataset.shape)
		
		
		

	print(np.array(train_dataset)[:,1])
	

	targets = np.array(train_dataset)[:,1]
	# inputs = 

	a = [1,1,23,1,23,1,24]

	random.shuffle(train_dataset)
	split_coeff = 0.2
	test_data = train_dataset[0:int(len(train_dataset)*0.2):1]
	train_dataset = train_dataset[int(len(train_dataset)*0.2)::1]
	print(len(test_data))
	print(len(train_dataset))
	
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
	test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size = batch_size,shuffle = True),
	loss_function = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)

	
	

	x = torch.randn(1,10,100,100)

	# exit()

	for epoch in range(0, 5): # 5 epochs at maximum
		print(f'Starting epoch {epoch+1}')
	
#     # Set current loss value
		current_loss = 0.0
	
    # Iterate over the DataLoader for training data
		for i, data in enumerate(train_loader, 0):
		
			# Get inputs
			print(i)
			inputs, targets = data
			# inputs=np.array([inputs])
			print(inputs.shape)
			inputs=inputs.reshape((1,10,100,100))
			print(inputs.shape)
			# print(inputs)
		# Zero the gradients
			optimizer.zero_grad()
		
		# Perform forward pass
			outputs = convnet(inputs)
		
		# Compute loss
			loss = loss_function(outputs, targets)
		
		# Perform backward pass
			loss.backward()
			optimizer.step()
			current_loss += loss.item()
		if i % 500 == 499:
			print('Loss after mini-batch %5d: %.3f' %(i + 1, current_loss / 500))
			current_loss = 0.0
	print('Training process has finished.')
		# Perform optimizationrandom.shuffle(train_dataset)

# 		# Perform forward pass
# 			outputs = convnet(inputs)
		
# 		# Compute loss
# 			loss = loss_function(outputs, targets)
		
# 		# Perform backward pass
# 			loss.backward()
		
# 		# Perform optimization
# 			optimizer.step()
		
# 		# Print statistics
# 		current_loss += loss.item()
# 		if i % 500 == 499:
# 			print('Loss after mini-batch %5d: %.3f' %
# 				(i + 1, current_loss / 500))
# 			current_loss = 0.0

# 		# Process is complete.
# 	print('Training process has finished.')
# 		current_loss += loss.item()
# 		if i % 500 == 499:
# 			print('Loss after mini-batch %5d: %.3f' %
# 				(i + 1, current_loss / 500))
# 			current_loss = 0.0

# 		# Process is complete.
# 	print('Training process has finished.')

  
