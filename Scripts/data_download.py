import os
import urllib.request
import random 
from tqdm import tqdm
from multiprocessing import Pool

file_list = 'binvox_file_locations.txt'
data_path = '../ShapeNet_Data/objects/'
http_path = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'
wanted_objects = ['plane']
max_num = 10000

# labels for the union of the core shapenet classes and the ikea dataset classes 
labels = {
	'03001627': 'chair', 
	'04379243': 'table',
	'02858304': 'boat',
	'02958343': 'car',  
	'02691156': 'plane', 
	'02808440': 'bathtub',
	'02871439': 'bookcase', 
	'02773838': 'bag',
	'02801938': 'basket',
	'02828884': 'bench',
	'02880940': 'bowl', 
	'02924116': 'bus',
	'02933112': 'cabinet',
	'02942699': 'camera',
	'03207941': 'dishwasher', 
	'03211117': 'display',
	'03337140': 'file',
	'03624134': 'knife',
	'03642806': 'laptop',
	'03710193': 'mailbox',
	'03761084': 'microwave',
	'03928116': 'piano',
	'03938244': 'pillow',
	'03948459': 'pistol',
	'04004475': 'printer', 
	'04099429': 'rocket',
	'04256520': 'sofa',
	'04554684': 'washer'
}

wanted_classes = []
for index in labels:
	if labels[index] in wanted_objects:
		wanted_classes.append(index)

if not os.path.exists(data_path):
	os.makedirs(data_path)


# download .obj files from shapenet
def download():
	with open(file_list, 'rb') as f:
		content = f.readlines()

	# make sub-directories for each class
	for i in wanted_classes:
		obj_path = data_path + labels[i] + '/'
		if not os.path.exists(obj_path):
			os.makedirs(obj_path)

	# search object for correct object classes
	obj_urls = []
	for filename in content:
		current = str(filename).split('/')		# './index/random/name'
		if current[1] in wanted_classes:
			if '_' in current[3]: continue			# redundent index
			if 'presolid' in current[3]: continue	# redundent index
			url_path = [http_path + current[1] + '/' + current[2] + '/model.obj',
			 			data_path + labels[current[1]] + '/' + current[2] + '.obj']
			obj_urls.append(url_path)

	# randomly sample objects of limited number
	obj_urls = random.sample(obj_urls, len(obj_urls))
	dictionary = {}
	final_urls = []
	for url in obj_urls:
		obj_class = url[1].split('/')[-2]
		if obj_class in dictionary:
			dictionary[obj_class] += 1
			if dictionary[obj_class] > max_num:
				continue
		else:
			dictionary[obj_class] = 1
		if os.path.exists(url[1]):
			continue
		final_urls.append(url)

	# parallel downloading of object .obj files
	pool = Pool()
	with tqdm(total = len(final_urls)) as pbar:
		for i, _ in tqdm(enumerate(pool.imap_unordered(down, final_urls))):
			pbar.update()


# simple function for parallel download processing
def down(url):
	urllib.request.urlretrieve(url[0], url[1])
	lines = ""
	with open(url[1], "r") as f:
		for line in f:
			if 'mtl' not in line:
				lines += line
	with open(url[1], "w") as f:
		f.write(lines)


if __name__ == '__main__':
	download()























