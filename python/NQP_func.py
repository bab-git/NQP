"""
================================
Non-Negative Quadratic Pursuit
================================

 The non-negative quadratic pursuit algorithm

 This code solves the following problem:

    minimize_x 0.5*x'*H*x + C'*x
         s.t   x>=0  , ||x||_0<T0

 The detail of the algorithm is described in the following paper:
 'Confident kernel dictionary learning for discriminative representation
  of multivariate time-series', B. Hosseini, F. Petitjean, Forestier G., and B. Hammer.

 Written by Babak Hosseini <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
 Copyright 2019 by Babak Hosseini

"""
print(__doc__)

import numpy as np
import sys
import scipy.sparse as sp

    
def f_x(H,C,x):
    cost = 0.5*x.T @ H @ x + np.dot(C,x)
    return cost

#===================

def NQP_func(H,C,T0):
#    print ("C=",C)
#    print ()
#    print ("H=",H)
#     print ("T=",T0)
    
    e,v = np.linalg.eig(H)
    if min(e.real)<0 and abs(min(e.real)/max(e.real))>1e-4:
        sys.exit('H is indifinite!')
    if np.linalg.norm(H-H.T, 'fro')>1e-4:
        sys.exit('H is not symmetric!')
    
    
    e_tol = 1e-6
    e_conv = 1e-3
    N = len(H)
    X = np.zeros(N)
    S = [] # positions indexes of components of s
    S0 = []
    R = list(range(N))
    L_1=0;
    i, =np.where(H.sum(0)==0)   # finding zero columns/rows in H to remove them from the set
    R.remove(i[0])          
    res_phi = C[:]
    res_x = 100
    x_est0 = np.zeros(N)
    #x_est=[];
    t = 1
    while t <= T0:
        i = np.where(res_phi[R]<-e_tol)   
        if len(i[0]) == 0:
            break
        
        jr = res_phi[R].argmin()  # most negative gradiant
        j = R[jr]
        S.append(j)
        R.remove(j)
        
        Hi = H[:,S]
        Hi = Hi[S,:]
        Ci=C[S]
        
        # Cholseky decomposition
        if t ==1:
            L = np.sqrt(Hi)
        else:
            v = H[S0,j]
            w = np.linalg.pinv(L_1) @ v
            c = H[j,j]
            if (c-np.dot(w,w)) <1e-4:  # dependant selection from H columns
                S.remove(j)
                continue    
            cx=np.zeros(1)
            cx[0]=np.sqrt(c-np.dot(w,w))
            wx=np.concatenate((w,cx)).T
            Lx=np.concatenate((L_1,np.zeros((len(L_1),1))),axis=1)
    #        L = np.concatenate((Lx.ravel(),wx))
            L = np.vstack((Lx, wx))
    #        sys.exit("write the code")
            
        Lp = np.linalg.pinv(L)
        x_est = -Lp.T @ (Lp @ Ci)
        
        # Find the first zero crossing        
    #    if sum((sign(x_est) < 0))      
        if sum(np.sign(x_est)) != len(x_est):
            neg_flag = 1
            x20 = np.zeros(len(H))
            x20[S0] = x_est0
            x20 = x20[S];       
            Hi2 = Hi[:]
            Ci2 = Ci[:]
            while 1:
                progress = (0 - x20)/(x_est - x20)
                i_n = np.where(x_est<0)
                p_neg=progress[i_n]
                i_p=np.argmin(p_neg)            
                remov=i_n[i_p]
                Hi2 = np.delete(Hi2,remov,axis=0)
                Hi2 = np.delete(Hi2,remov,axis=1)            
                Ci2 = np.delete(Ci2,remov) 
                del S[remov[0]]
                
                L = np.linalg.cholesky(Hi2)   # Cholesky factorization
                Lp = np.linalg.pinv(L)
                x_est = -Lp.T @ (Lp @ Ci2)
                
                x_est[x_est<1e-5]=0
                if sum(np.sign(x_est)) != len(x_est):
                    x20[remov]=[];
                else:
                    break   
            Hi = Hi2 [:]
            Ci = Ci2 [:]
        else:
            neg_flag = 0
            
        # Checking the quality of the selected column
        res_x0=f_x(Hi,Ci,x_est)
        if np.linalg.norm(res_x0-res_x)/np.linalg.norm(res_x) > e_tol and neg_flag ==0:    
            res_x=res_x0
        elif neg_flag == 0:
            S.remove(j)
            if len(S) == 0:
                x_est =0
            else:
                L = L_1[:]
                x_est = x_est0        
        else:
            res_x=res_x0 # compare next round to current zero-crossing x
        
        if len(S) == 0:
            res_phi = C[:]
        else:
            res_phi = C.ravel() + (H[:,S] @ x_est).ravel()
        
        S0 = S[:]
        x_est0 = x_est[:] 
        L_1 = L[:]
        if sum(x_est) > 0:
            t = x_est.size+1;
        
        if abs(res_x) < e_conv:
            print('The minimum objective threshold is reached.')
            print('Change e_conv if you want a smaller objective value.')    
        if len(R) ==0:
            break   
    
    
    X = np.zeros(N)
    if len(x_est) ==0:
        print('No valid solution, please check the validity of H and C')
    else:
        X[S] = x_est[:]
        print('Optimization is Convereged')
    #    print()
    X[X/max(X)<1e-2] = 0
    fobj = f_x(H,C,X)
    X = sp.csr_matrix(X)
#    print ("X=",X)
#    print(X)
#    print ("fobj = %8.2f"% fobj)
    
    return X, fobj

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
