
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
    n_update_sample = 10
    dim = 5
    n_cluster = 8
    
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


def PrintTree(tree,X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    estimator=tree
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        i = int(i)
        if is_leaves[i]:
            print i,"leaf"
        else:

            print i,"node"
    print()


def splitLeaf(tree, X, y):
    """
    split a leaf node with id `node_id` into two new leaf nodes.
    and return the new leaf_ids.
    """

    tree2 = DecisionTreeRegressor(random_state=tree.random_state,min_samples_leaf=tree.min_samples_leaf)
    '''
    min_weight_leaf = (tree.min_weight_fraction_leaf *
                   len(X))
    max_depth = ((2 ** 31) - 1 if tree.max_depth is None
                     else tree.max_depth)
    random_state = sklearn.utils.check_random_state(tree.random_state)
    criterion = sklearn.tree._criterion.MSE(tree.n_outputs_, len(X))
    splitter = sklearn.tree._splitter.BestSplitter(criterion,
                                                tree.max_features_,
                                                tree.min_samples_leaf,
                                                min_weight_leaf,
                                                random_state,
                                                tree.presort)
    from sklearn.tree._tree import DepthFirstTreeBuilder

    new_node_id = tree.tree_._add_node(None, True, True, 0,
                                         0.5, tree.min_impurity_split, 2,
                                         None)
    #print "new_node_id",new_node_id
    
    #builder = DepthFirstTreeBuilder(splitter, tree.min_samples_split,
    #                                        tree.min_samples_leaf,
    #                                        min_weight_leaf,
    #                                        max_depth, tree.min_impurity_split)
    #builder.build(tree.tree_, X, y, None, None)
    return tree
    '''
    return tree2.fit(X,y)
         

#test decision tree recursive growth
from sklearn.tree import DecisionTreeRegressor
import sklearn
tree = DecisionTreeRegressor(random_state=41,min_samples_leaf=1)
X = np.random.uniform(0.0,1.0,(4,2))
y = np.random.uniform(0.0,1.0,4)
tree = tree.fit(X, y)
#PrintTree(tree,X,y)
addX = [-0.1,-.1]
appliedx = tree.apply(addX)
print "node",appliedx
newX = np.vstack((X,addX))
newY = np.append(y, -1.1).reshape(-1,1)
print newX.shape
print newY.shape
print appliedx
tree = splitLeaf(tree, newX, newY)
appliedx = tree.apply(newX)
print "new node",appliedx

#PrintTree(tree,newX,newY)



exit()

 

for method in [ 'tree', 'k-mean','random','OK']:
    import time
    start = time.time()
    runtest(method)
    print "| METHOD",method,"DONE IN",time.time() - start
