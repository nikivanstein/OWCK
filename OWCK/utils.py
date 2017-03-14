# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:28 2015

@author: wangronin
"""

import time, os
import numpy as np
from numpy import pi, log, atleast_2d, size, mod
#from pyDOE import lhs
#from ghalton import Halton
#from sobol import i4_sobol

## SMSE measurement
# test_y is the target, pred_y the predicted target, both 1D arrays of same length
def SMSE(test_y,pred_y):
	se = []
	target_variance = np.var(test_y)
	for i in range(len(test_y)):
		temp = (pred_y[i] - test_y[i])**2
		se.append(temp)
	mse = np.mean(se)
	smse = mse / target_variance
	return smse

## MSLL = mean standardized log loss
## logprob = 0.5*log(2*pi.*varsigmaVec) + sserror - 0.5*log(2*pi*varyTrain)...
##           - ((yTestVec - meanyTrain).^2)./(2*varyTrain);
def MSLL(train_y,test_y,pred_y,variances):
	sll = []
	mean_y = np.mean(train_y)
	var_y = np.var(train_y)
	for i in range(len(variances)):
		if variances[i] == 0:
			variances[i] += 0.0000001 #hack
		sll_trivial = 0.5*log(2 * pi * var_y) + ((test_y[i] - mean_y)**2 / (2* var_y)) 
		sllv = ( 0.5*log(2 * pi * variances[i]) + \
      ((test_y[i] - pred_y[i])**2 / (2* variances[i])) ) - sll_trivial
		sll.append(sllv)
	sll = np.array(sll)
	msll = np.mean(sll)
	return msll
    

# # Obtain the initial design locations
# def get_design_sites(dim, n_sample, x_lb, x_ub, sampling_method='lhs'):
    
#     x_lb = atleast_2d(x_lb)
#     x_ub = atleast_2d(x_ub)
    
#     x_lb = x_lb.T if size(x_lb, 0) != 1 else x_lb
#     x_ub = x_ub.T if size(x_ub, 0) != 1 else x_ub
    
#     if sampling_method == 'lhs':
#         # Latin Hyper Cube Sampling: Get evenly distributed sampling in R^dim
#         samples = lhs(dim, samples=n_sample) * (x_ub - x_lb) + x_lb
        
#     elif sampling_method == 'uniform':
#         samples = np.random.rand(n_sample, dim) * (x_ub - x_lb) + x_lb
        
#     elif sampling_method == 'sobol':
#         seed = mod(int(time.time()) + os.getpid(), int(1e6))
#         samples = np.zeros((n_sample, dim))
#         for i in range(n_sample):
#             samples[i, :], seed = i4_sobol(dim, seed)
#         samples = samples * (x_ub - x_lb) + x_lb
        
#     elif sampling_method == 'halton':
#         sequencer = Halton(dim)
#         samples = sequencer.get(n_sample) * (x_ub - x_lb) + x_lb
        
#     return samples
