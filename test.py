
import time, sys, pdb
import numpy as np
from numpy import ones
from numpy.random import rand
from sklearn import cross_validation

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from deap import benchmarks

from OWCK import OWCK, GaussianProcess_extra

from pyDOE import lhs
from ghalton import Halton
from sobol import i4_sobol


def get_design_sites(dim, n_sample, x_lb, x_ub, sampling_method='lhs'):
    x_lb = np.atleast_2d(x_lb)
    x_ub = np.atleast_2d(x_ub)
    
    x_lb = x_lb.T if np.size(x_lb, 0) != 1 else x_lb
    x_ub = x_ub.T if np.size(x_ub, 0) != 1 else x_ub
    
    if sampling_method == 'lhs':
        # Latin Hyper Cube Sampling: Get evenly distributed sampling in R^dim
        samples = lhs(dim, samples=n_sample) * (x_ub - x_lb) + x_lb
        
    elif sampling_method == 'uniform':
        samples = np.random.rand(n_sample, dim) * (x_ub - x_lb) + x_lb
        
    elif sampling_method == 'sobol':
        seed = mod(int(time.time()) + os.getpid(), int(1e6))
        samples = np.zeros((n_sample, dim))
        for i in range(n_sample):
            samples[i, :], seed = i4_sobol(dim, seed)
        samples = samples * (x_ub - x_lb) + x_lb
        
    elif sampling_method == 'halton':
        sequencer = Halton(dim)
        samples = sequencer.get(n_sample) * (x_ub - x_lb) + x_lb
        
    return samples

#test function to model
def ackley_arg0(sol):
    
    X,Y = sol[0], sol[1]
    Z = np.zeros(X.shape)
    
    for i in xrange(X.shape[0]):
        Z[i] = benchmarks.rastrigin((X[i],Y[i]))[0]
    return Z


def runtest(cluster_method='k-mean'):
    
    d = lambda:0 
    n_sample = 1000
    n_update_sample = 500
    dim = 5
    n_cluster = 10
    
    x_lb = np.array([-29.9] * dim)
    x_ub = np.array([29.9] * dim)
    
    X = get_design_sites(dim, n_sample, x_lb, x_ub, 'lhs')

    d.data = X
    d.target = ackley_arg0(d.data.T)

    X_update = get_design_sites(dim, n_update_sample, x_lb, x_ub, 'lhs')
    Y_update = ackley_arg0(X_update.T)

	#STANDARDIZE DATA
    #std_scaler = StandardScaler()
    #d.data = std_scaler.fit_transform(d.data ,y=d.target)
    #X_update = std_scaler.transform(X_update,y=Y_update)
    
    #std_scaler = StandardScaler(with_std=False)
    #d.target = std_scaler.fit_transform(d.target)
    #Y_update = std_scaler.transform(Y_update)

    seed_times = 3
    
    kf = cross_validation.KFold(len(d.target), n_folds=seed_times)
    
    seed = 0
    
    # bounds of hyper-parameters for optimization
    thetaL = 1e-5 * ones(dim)  * 29.9
    thetaU = 100 * ones(dim)  * 29.9
    
    # initial search point for hyper-parameters
    theta0 = rand(1, dim) * (thetaU - thetaL) + thetaL
    
    for train_index, test_index in kf:
        
        seed += 1
        
        #print "FOLD: ", seed

        time_run_start = time.time()
        if (method=='OK'):

            owck_model = GaussianProcess_extra(regr='constant', corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU, nugget=None, verbose=False, nugget_estim=True, random_start = 100*dim)
            owck_model.fit(d.data[train_index], d.target[train_index])
        else:
            owck_model = OWCK(regr='constant', corr='matern', cluster_method=cluster_method, overlap=0.1, theta0=theta0, thetaL=thetaL, thetaU=thetaU, 
                              n_cluster=int(n_sample/100), nugget=None, verbose=False,
                              nugget_estim=True, random_start = 100*dim,
                              is_parallel=False)
            owck_model.fit(d.data[train_index], d.target[train_index])
            #update the model with additional data (just testing)

            owck_model.fit(X_update,Y_update)

        
        time_run = time.time() - time_run_start
        #sys.stderr.write("--- Fitting finished in "+`time_run`+" ---\n" )
        
        time_left = time_run * (seed_times-seed-1)
        time_left_hour = int(time_left / 3600)
        time_left_minute = int((time_left - 3600*time_left_hour)/60)
        #sys.stderr.write("--- Aproximate time left (HH:MM): "+`time_left_hour`+":"+`time_left_minute`+" ---\n" )

        predictions = owck_model.predict(d.data[test_index])
        predictions = np.array(predictions).reshape(-1, 1)
        score = r2_score(d.target[test_index], predictions)
        print cluster_method,"R2 score:", score


 

for method in [ 'tree', 'k-mean','random','OK']:
    import time
    start = time.time()
    runtest(method)
    print "| METHOD",method,"DONE IN",time.time() - start
