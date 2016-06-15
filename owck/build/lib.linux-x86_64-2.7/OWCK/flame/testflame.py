#test flame
import ctypes
import flame
import numpy as np
import numpy.ctypeslib as npct


#data = np.array([[0,1],[1,0],[10.1,10.2],[10.3,10.4]]).astype(np.float32) #np.random.rand(1000,10).astype(np.float32, copy=False)
data  = np.loadtxt("testmatrix.txt").astype(np.float32)

#data[500:] = data[500:]*1000
N = len(data)
M = len(data[0])





#print data
#print float_array

print data.shape
flameobject = flame.Flame_New()
flame.Flame_SetDataMatrix( flameobject, data,  0 )

print "Detecting Cluster Supporting Objects ..."

flame.Flame_DefineSupports( flameobject, 10, -2.0 )

cso_count = flameobject.cso_count 
print "done, found ", cso_count

print "Propagating fuzzy memberships ... " 

flame.Flame_LocalApproximation( flameobject, 500, 1e-6 )

cso_count = flameobject.cso_count 
print "done, found ", cso_count
print "done"

print "Defining clusters from fuzzy memberships ... " 

flame.Flame_MakeClusters( flameobject, -1.0 )
#print "done"

fyzzyzooi = flame.Print_Clusters(flameobject, (cso_count+1)*N )
flame.Flame_Clear(flameobject)
fyzzyzooi = fyzzyzooi.reshape(( N,cso_count+1 ))
print fyzzyzooi
exit()

#todo fix it later



