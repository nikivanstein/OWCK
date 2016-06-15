
import time, sys
import numpy as np
from numpy import ones
from numpy.random import rand
from sklearn import cross_validation

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from deap import benchmarks

from OWCK import OWCK
from OWCK.utils import get_design_sites

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
    n_update_sample = 1000
    dim = 2
    n_cluster = 8
    
    x_lb = np.array([-29.9] * dim)
    x_ub = np.array([29.9] * dim)
    
    X = get_design_sites(dim, n_sample, x_lb, x_ub, 'lhs')

    d.data = X
    d.target = ackley_arg0(d.data.T)

    X_update = get_design_sites(dim, n_update_sample, x_lb, x_ub, 'lhs')
    Y_update = ackley_arg0(X_update.T)

	#STANDARDIZE DATA
    std_scaler = StandardScaler()
    d.data = std_scaler.fit_transform(d.data ,y=d.target)
    X_update = std_scaler.transform(X_update,y=Y_update)
    
    std_scaler = StandardScaler(with_std=False)
    d.target = std_scaler.fit_transform(d.target)
    Y_update = std_scaler.transform(Y_update)

    seed_times = 5
    
    kf = cross_validation.KFold(len(d.target), n_folds=seed_times)
    
    seed = 0
    
    # bounds of hyper-parameters for optimization
    thetaL = 1e-5 * ones(dim) 
    thetaU = 10 * ones(dim) 
    
    # initial search point for hyper-parameters
    theta0 = rand(1, dim) * (thetaU - thetaL) + thetaL
    
    for train_index, test_index in kf:
        
        seed += 1
        
        print "FOLD: ", seed

        time_run_start = time.time()

        owck_model = OWCK(regr='constant', corr='squared_exponential', cluster_method=cluster_method, overlap=0.1,
                          theta0=theta0, thetaL=thetaL, thetaU=thetaU, 
                          n_cluster=n_cluster, nugget=1e-8, verbose=False,
                          is_parallel=True)

        owck_model.fit(d.data[train_index], d.target[train_index])
        #update the model with additional data (just testing)
        owck_model.updateModel(X_update,Y_update)

        time_run = time.time() - time_run_start
        sys.stderr.write("--- Fitting finished in "+`time_run`+" ---\n" )
        
        time_left = time_run * (seed_times-seed-1)
        time_left_hour = int(time_left / 3600)
        time_left_minute = int((time_left - 3600*time_left_hour)/60)
        sys.stderr.write("--- Aproximate time left (HH:MM): "+`time_left_hour`+":"+`time_left_minute`+" ---\n" )

        predictions, variance = owck_model.predict(d.data[test_index])
        score = r2_score(d.target[test_index], predictions)
        print cluster_method,"R2 score:", score

for method in ['tree','k-mean', 'GMM', 'fuzzy-c-mean', 'flame', 'random']:
    runtest(method)
