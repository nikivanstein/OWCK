The Optimal Weighted Cluster Kriging/Gaussian Process class
=======================

This class inherited from GaussianProcess class in sklearn library
Most of the parameters are contained in sklearn.gaussian_process.

Please check the docstring of Gaussian Process parameters in sklearn.
Only newly introduced parameters are documented below.

Install Instructions
================

Just run the install.py or install directly with pip.
You need OpenMPI installed to use the parralel options.

Pip::

    pip install OWCK

Parameters
----------
n_cluster : int, optional
    The number of clusters, determines the number of the Gaussian Process
    model to build. It is the speed-up factor in OWCK.
cluster_method : string, optional
    The clustering algorithm used to partition the data set.
    Built-in clustering algorithm are:
        'k-mean', 'GMM', 'fuzzy-c-mean', 'random', 'tree'
        Note that GMM, fuzzy-c-mean are fuzzy clustering algorithms 
        With these algorithms you can set the overlap you desire.
        tree is a regression tree clustering-based approach
overlap : float, optional
    The percentage of overlap when using a fuzzy cluster method.
    Each cluster will be of the same size.
is_parallel : boolean, optional
    A boolean switching parallel model fitting on. If it is True, then
    all the underlying Gaussian Process model will be fitted in parallel,
    supported by MPI. Otherwise, all the models will be fitted sequentially.
    
Attributes
----------
cluster_label : the cluster label of the training set after clustering
clusterer : the clustering algorithm used.
models : a list of (fitted) Gaussian Process models built on each cluster.

Usage
----------
Example code::

    from OWCK import OWCK
    owck_model = OWCK(cluster_method='tree')
    owck_model.fit(X,y)
    pred_y, var_y = owck_model.predict(x_new)

References
----------

.. [SWKBE15] `Bas van Stein, Hao Wang, Wojtek Kowalczyk, Thomas Baeck 
    and Michael Emmerich. Optimally Weighted Cluster Kriging for Big 
    Data Regression. In 14th International Symposium, IDA 2015, pages 
    310-321, 2015`
    http://link.springer.com/chapter/10.1007%2F978-3-319-24465-5_27#
