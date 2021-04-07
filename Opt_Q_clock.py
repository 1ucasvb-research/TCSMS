import numpy as np
import numba as nb
import time
from numba import jit, njit, prange, float64, int32, int64, complex64, complex128,  types


#Estimated q, it doesn't seem that important, also 1/4 was OK.
#Notice that this q corresponds to sqrt(q) of the main text
@njit('float64(int32)')
def gen_q(d):
    #return 1/4 
    return np.sqrt(1-d/(d+1))


@njit('float64(int32)')
def gen_theta(d):
	return (np.pi*2/d) * (1-1/d)



#Root of unity
@njit('complex128(int32,int32)')
def omega(a,d):
    return np.exp(2*1j*np.pi*a/d)

#Discrete Fourier Transform
@njit('complex128[:,:](int32)')
def DFT(d):
    F= np.zeros((d,d), dtype=np.cdouble)
    for i in range(d):
        for j in range(d):
            F[i,j]=omega(i*j,d)

    F = F/np.sqrt(d)
    
    return F

@njit('complex128[:,:](float64,int32)')
def gen_U(theta,d):
    U=np.zeros((d,d), dtype=np.cdouble)
    for i in range(d):
        U[i,i]=np.exp(-1j*i*theta)

    F = DFT(d)
    U = F@U@F.conj().T
    return U

#Generate Sqrt(E0) 
@njit('complex128[:,:](int32)')
def gen_E(d):
    E = np.eye(d, dtype=np.cdouble)
    q = gen_q(d)
    E[-1,-1]= q
    return E
    
#Generate Kraus operator K0
@njit('complex128[:,:](float64,int32)')
def gen_kraus(theta,d):
    U= gen_U(theta,d)
    E = gen_E(d)
    K = U@E
    
    return K

@njit('float64(float64,int32)')
def comp_prob(theta,d):

    K = gen_kraus(theta,d)
    Kd = K
    for i in range(d-1):
        Kd = Kd@K

    q = gen_q(d)
    pL = (1-q**2)*np.abs(Kd[-1,0])**2 ## p=(0^d 1) = (1-q^2) |<d-1| K0^d | 0>|^2 for E0 = diag(1,..., q^2)


    return pL
    


#Parallelized  verison
@njit('types.Tuple((float64[:], float64[:]))(int32,int32)', parallel=True)
def gen_plot_points(din,dfin):
    num_points=dfin-din+1
    p = np.zeros(num_points)
    th = np.zeros(num_points)
    for j in prange(num_points):
        if din+j % 100 == 0:
            print("Computing for dimension", din+j)
        th[j] = gen_theta(din+j)
        p[j] = comp_prob(th[j],din+j)
        

    

    return p, th

        

#######################################################################

#########################   MAIN   ####################################

#######################################################################


maxp = []
optth = []
#Initial and final dimensions
din = 2
dfin = 500

tic = time.time()

p, th = gen_plot_points(din,dfin)

toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))



outfile='plot_'+str(din)+'_'+str(dfin)+'.dat'
header = 'Result plot from' + str(din) + 'up to d: ' + str(dfin) + '\n# Prob vector\n'

#res = np.concatenate((p,th), axis=0)


outfilep='new_prob_'+str(din)+'_'+str(dfin)+'.dat'
p.astype('float32').tofile(outfilep)


#outfileth='th_'+str(din)+'_'+str(dfin)+'.dat'
#th.astype('float32').tofile(outfileth)



