"""
This is the demo use of the Non-negatice Quadratic Pursuit algorithm
Written by Babak Hosseini <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
Copyright 2019 by Babak Hosseini
"""
print(__doc__)

import numpy as np
from NQP_func import NQP_func

n = 20    # dimension of H
T0 = 5    # sparsity limit
H = np.random.rand(n,n)
H [3,:] = 0
H = H @ H.T
#H = np.loadtxt('Hmatrix.csv',delimiter=',')
C = np.random.rand(n)-0.5
#C = np.loadtxt('Cmatrix.csv',delimiter=',')

#print ("C=",C)
#print ("H=",H)

X,fobj = NQP_func(H,C,T0)
print ()
print ("X=\n",X)
print ("objctive-value = %8.2f"% fobj)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
