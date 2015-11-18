# OWCK
## Optimal Weighted Cluster Kriging/Gaussian Process for Python

    
This class inherited from GaussianProcess class in sklearn library
Most of the parameters are contained in sklearn.gaussian_process.

Please check the docstring of Gaussian Process parameters in sklearn.
Only newly introduced parameters are documented below.

### Parameters
----------
 - n_cluster : int, optional
    The number of clusters, determines the number of the Gaussian Process
    model to build. It is the speed-up factor in OWCK.
 - cluster_method : string, optional
    The clustering algorithm used to partition the data set.
    Built-in clustering algorithm are:
        'random', 'k-mean'
 - is_parallel : boolean, optional
    A boolean switching parallel model fitting on. If it is True, then
    all the underlying Gaussian Process model will be fitted in parallel,
    supported by MPI. Otherwise, all the models will be fitted sequentially.
    
### Attributes
----------
 - cluster_label : the cluster label of the training set after clustering
 - clusterer : the clustering algorithm used.
 - models : a list of (fitted) Gaussian Process models built on each cluster.

### References
----------

- [SWKBE15] `Bas van Stein, Hao Wang, Wojtek Kowalczyk, Thomas Baeck 
    and Michael Emmerich. Optimally Weighted Cluster Kriging for Big 
    Data Regression. In 14th International Symposium, IDA 2015, pages 
    310-321, 2015`
    http://link.springer.com/chapter/10.1007%2F978-3-319-24465-5_27#