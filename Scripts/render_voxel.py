import os
import sys
import shutil

import urllib.request
import random 
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from subprocess import call

dimension = 256
obj_path = '../ShapeNet_Data/objects/'
vox_path = '../ShapeNet_Data/binvox' + str(dimension) + '/'
wanted_objects = ['chair']


if not os.path.exists(vox_path):
	os.makedirs(vox_path)


debug_mode = 1 # change to make all of the called scripts print their errors and warnings 
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'


# converts .obj files to .binvox files, intermidiate step before converting to voxel .npy files 
def binvox():
	for s in wanted_objects:
		dirs = glob(obj_path + s + '/*.obj')
		commands =[]
		count = 0 
		for filepath in tqdm(dirs):
			command = 'binvox.exe ' + filepath  + ' -d ' + str(dimension) + ' -pb -cb -c -dc -aw -e'   # this executable can be found at http://www.patrickmin.com/binvox/ , 
			# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight 
			commands.append(command)
			if count % 1000 == 0  and count != 0:
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count += 1 

		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()


def call(command):
	os.system('%s %s' % (command, io_redirect))


def movefiles():
	for s in wanted_objects:
		if not os.path.exists(vox_path + s):
			os.makedirs(vox_path + s)
		dirs = glob(obj_path + s + '/*.binvox')
		for filepath in tqdm(dirs):
			filename = filepath.split('\\')[1].split('.')[0]
			srcfile = obj_path + s + '/' + filename + '.binvox'
			dstfile = vox_path + s + '/' + filename + '.binvox'
			shutil.move(srcfile, dstfile)


def confirm():
	print('confirm message:')
	for s in wanted_objects:
		dirs = glob(obj_path + s + '/*.obj')
		print('obj numbers: ', len(dirs))
		dirs2 = glob(vox_path + s + '/*.binvox')
		print('vox numbers:', len(dirs2))
		if len(dirs) != len(dirs2):
			for i in dirs:
				flag = False
				name = i.split('\\')[1].split('.')[0]
				for j in dirs2:
					name2 = j.split('\\')[1].split('.')[0]
					if (name == name2):
						flag = True
						break
				if flag == False:
					print(name)


if __name__ == '__main__':
	binvox()
	movefiles()
	confirm()
