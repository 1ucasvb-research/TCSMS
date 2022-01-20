import os, sys
import numpy as np
import random
from numba import jit, njit, prange, float64, complex128, int64, types

GRAD_STEP = 1e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

NUM_TRIALS = 30
MAX_IT = 5000
LR = 0.003

# -----------------------------------------------------------------

def number_to_base(n, b, p=0):
	if n == 0:
		return np.array([0]*(p - 1) + [0], dtype=np.int64)
	digits = []
	while n:
		digits.append(int(n % b))
		n //= b
	return np.array([0]*(p - len(digits)) + digits[::-1], dtype=np.int64)

# N = number of operators for I0\I1, d = dimension
# Total of 2N
# [2N, Re/Im, d, d]
@njit("float64[:,:,:,:](int64, int64)", cache=True)
def random_B(n, d):
	return (2.0*np.random.random((2*n, 2, d, d)) - 1.0).astype(float64)

# @njit("types.Tuple((complex128[:,:,:], float64))(float64[:,:,:,:])", cache=True)
@njit("complex128[:,:,:](float64[:,:,:,:])", cache=True)
def get_Kraus(B):
	nt, d = B.shape[0], B.shape[2]
	E = np.zeros((d, d), dtype=complex128)
	Ks = np.zeros((nt, d, d), dtype=complex128)
	for k in range(nt):
		Ks[k] = B[k,0] + 1j*B[k,1]
		E += Ks[k].conj().T @ Ks[k]
	l = np.linalg.eigvalsh(E)[-1]
	# return Ks / np.sqrt(l), np.real(np.trace(E)) / (l*d)
	return Ks / np.sqrt(l)

@njit("float64(float64[:,:,:,:], int64, int64, int64[:])", cache=True)
def compute_probability(B, n, d, seq):
	Ks = get_Kraus(B)
	K = Ks[0:n], Ks[n:]
	y = np.eye(d, dtype=complex128)
	for s in seq[::-1]:
		x = np.zeros((d,d), dtype=complex128)
		for i in range(0, n):
			k = np.copy(K[s][i]) # we copy the matrix, getting rid of the view, so it becomes contiguous
			x += k.conj().T  @ y @ k
		y = x
	return -np.real(y[0,0])

@njit("float64[:,:,:,:](float64[:,:,:,:], int64, int64, int64[:])", cache=True)
def compute_gradient(B, n, d, seq):
	G = np.zeros((2*n, 2, d, d), dtype=float64)
	H = np.zeros((2*n, 2, d, d), dtype=float64)
	for i in range(4*n*d**2):
		H.flat[i] = GRAD_STEP
		G.flat[i] = (compute_probability(B + H, n, d, seq) - compute_probability(B - H, n, d, seq)) / (2.0*GRAD_STEP)
		H.flat[i] = 0.0
	return G

@njit('types.Tuple((float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:]))(float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], int64, float64)', cache=True)
def adam_iteration(grad, m, v, t, lr):
	m2 = ADAM_BETA1*m +(1 - ADAM_BETA1)*grad
	v2 = ADAM_BETA2*v +(1 - ADAM_BETA2)*grad**2
	m_h = m2/(1 - ADAM_BETA1**(t + 1))
	v_h = v2/(1 - ADAM_BETA2**(t + 1))
	return lr*m_h/(np.sqrt(v_h) + ADAM_EPSILON), m2, v2 

@njit('float64[:,:,:,:](float64[:,:,:,:], float64, int64, int64, int64, int64[:])', cache=True)
def adam_optimize(B, lr, num_iter, n, d, seq):
	m = np.zeros((2*n, 2, d, d), dtype=float64)
	v = np.zeros((2*n, 2, d, d), dtype=float64)
	Bo = np.copy(B)
	for t in range(num_iter):
		grad = compute_gradient(Bo, n, d, seq)
		step, m, v = adam_iteration(grad, m, v, t, lr)
		Bo -= step
	return Bo

def full_optimize(n, d, seq):
	best_p, best_B = 0, None
	for it in range(NUM_TRIALS):
		B = adam_optimize(random_B(n, d), LR, MAX_IT, n, d, seq)
		p = -compute_probability(B, n, d, seq)
		print("\tTrial=%d/%d" % (it, NUM_TRIALS), "p=%0.06f" % p)
		if p > best_p:
			best_p = p
			best_B = np.copy(B)
	print("\nBest p=%0.06f" % best_p)
	return best_p, best_B

def check_kraus(B):
	Ks = get_Kraus(B)
	for k in Ks:
		print(k, "\n")
	E = np.zeros((D,D), dtype=np.complex128)
	for k in range(2*N):
		E += Ks[k].conj().T @ Ks[k]
	print(np.round(E,3))

def DC(seq):
	L = len(seq)
	dc = 0
	found = False
	while not found:
		dc += 1
		for l1 in range(dc):
			l2 = dc - l1
			match = True
			for i in range(L - dc):
				if seq[dc+i] != seq[l1+(i % l2)]:
					match = False
					break
			if match:
				found = True
	return dc

# ------------------------------------------------------------------------------

# This block of code generates a list of "cases" we want to check,
# shuffles them, and separates these into 8 separate batches that
# can be run in parallel in multiple instances on multiple machines.

# This was done to speed things up, as a proper multi threaded code
# would require more effort to implement. I figured I would keep it
# simple even if not elegant. :)

# D = 2
# N = 1
# L = 5
# dat = open(os.path.join("cases.txt"), "w")
# cases = []
# for L in [3,4,5,6,7]:
	# for seqi in range(1, 2**(L-1)):
		# seq = number_to_base(seqi, 2, L)
		# dc = DC(seq)
		# for d in range(1, dc):
			# for N in [1,2,3]:
				# cases.append([L, seqi, d, N])

# random.shuffle(cases)
# cases = np.array(cases, np.int64)

# for i in range(0,9):
	# np.save("cases%d.npy" % i,
		# cases[i * 127:(i+1)*127]
	# )


# These batches can then be executed by passing the batch number 
# to the qopt.py file as an argument
casen = int(sys.argv[1])
cases = np.load("cases%d.npy" % casen)
dat = open(os.path.join("probs%d.txt" % casen), "w")

for L, seqi, d, N in cases:
	seq = number_to_base(seqi, 2, L)
	dc = DC(seq)
	print("\nSequence:", seq, "DC=%d" % dc, "d=%d N=%d" % (d, N))
	best_p, best_B = full_optimize(N, d, seq)
	Ks = get_Kraus(best_B)
	np.save(os.path.join(
		"kraus",
		"kraus_L=%02d_seq=%s_n=%02d_d=%02d.npy" % (L, "".join(map(str, seq)), N, d)
		),
		np.stack([np.real(Ks).astype(np.float64), np.imag(Ks).astype(np.float64)])
	)
	dat.write("%d\t%d\t%d\t%d\t%f\n" % (
		L, seqi, d, N, best_p
	))
	dat.flush()

# EOF