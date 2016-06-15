# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:33:49 2015

@author: wangronin
"""


from mpi4py import MPI

comm = MPI.Comm.Get_parent()
comm_self = MPI.COMM_WORLD

model = comm.scatter(None, root=0)
data = comm.scatter(None, root=0)

index, training_set = data

while True:  
    try:
        model.fit(*training_set)
        break
    except ValueError:
        model.nugget *= 10  # TODO: need to discuss this simple fix!

# Synchronization with master process...
comm.Barrier()

# Gathering the fitted kriging model back
fitted = {
          'index': index, 
          'model': model
          }
          
comm.gather(fitted, root=0)

comm.Disconnect()