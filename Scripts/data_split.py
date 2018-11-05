import os
import random
import numpy as np
import pandas as pd

train_path = '../ShapeNet_Data/sets/train/'
valid_path = '../ShapeNet_Data/sets/valid/'
test_path = '../ShapeNet_Data/sets/test/'
data_path = '../ShapeNet_Data/objects/'

wanted_objects = ['chair', 'table', 'plane']

if not os.path.exists(train_path):
	os.makedirs(train_path)

if not os.path.exists(valid_path):
	os.makedirs(valid_path)

if not os.path.exists(test_path):
	os.makedirs(test_path)


for objclass in wanted_objects:
	classdir = data_path + objclass
	files = os.listdir(classdir)
	files = random.sample(files, len(files))
	train_set = np.array( files[:int(len(files)*0.7)] )
	valid_set = np.array( files[int(len(files)*0.7):int(len(files)*0.8)] )
	test_set  = np.array( files[int(len(files)*0.8):] )

	with open(train_path + objclass + '.txt', 'w') as f:
		for i in range(len(train_set)):
			filename = train_set[i].split('.')[0]
			f.write(filename + '\n')

	with open(valid_path + objclass + '.txt', 'w') as f:
		for i in range(len(valid_set)):
			filename = valid_set[i].split('.')[0]
			f.write(filename + '\n')

	with open(test_path + objclass + '.txt', 'w') as f:
		for i in range(len(test_set)):
			filename = test_set[i].split('.')[0]
			f.write(filename + '\n')

	print(objclass)
	print('train: ', len(train_set))
	print('valid: ', len(valid_set))
	print('test: ', len(test_set))


# chair 4232 604 1210
# table 4867 695 1391
# plane 2800 400 800