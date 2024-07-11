# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem, Peter Bener and Christian Lessig. All rights reserved.
# License: Apache-2.0

import argparse
import numpy as np
import struct

def compare_tensors(file1, file2):
	'''
	expects tensors of the form (N, k, 3)
	where k might be different for the given files

	'''
	a = np.load(file1, mmap_mode='r') 
	b = np.load(file2, mmap_mode='r')
	
	if a.shape[0]==b.shape[1] and a.shape[1]==b.shape[0] and a.shape[2]==b.shape[2]:
		a = np.swapaxes(a, 0, 1)
	if a.shape == b.shape:
		print('File one contains a ' +str(a.shape)+ ' tensor, and file two '+str(b.shape))
		print ('checking if identical ...', np.array_equal(a, b) )
		print ('checking if close ...', np.allclose(a, b) )
		
		testSparsity('a', a, 2)
		testSparsity('b', b, 2)
		
	elif a.shape[1] < b.shape[1]:
		print('File one contains a ' +str(a.shape)+ ' tensor, and file two '+str(b.shape))
		print ('checking if identical until ' + str(a.shape[1])+ ' bases ...', np.array_equal(a, b[: , :a.shape[1], :]) )
		
	else:
		print('File one contains a ' +str(a.shape)+ ' tensor, and file two '+str(b.shape))
		print ('checking if identical until ' + str(b.shape[1])+ ' bases ...', np.array_equal(a[: , :b.shape[1], :], b) )

def testSparsity(name, mat, test_dim):
	print(" testing the sparsity of " + name)
	sparPerList = []
	for l in range(mat.shape[test_dim]):
		sparPer = 1 - np.count_nonzero(mat[:, :, l])/ mat[:, :, l].size
		sparPerList.append(sparPer)
	if max(sparPerList) > 0.5:
		print(" ... sparse, with min %" + str(min(sparPerList)*100) + " zero entries in each dimension")
	else:
		print(" ... not sparse.")
		
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Compare bases stored in different .npy files')
	parser.add_argument('input_file_1')
	parser.add_argument('input_file_2')

	args = parser.parse_args()
	compare_tensors(args.input_file_1, 
		args.input_file_2)
