"""
Optimize probability for a given sequence via different optimization algorithms
included in pytorch package 
"""

import torch
import numpy as np
from math import *
import os

PATH = os.path.join("surveydata","classical")

# number of iterations
MAX_ITERATIONS           = 10000
# When to stop
TARGET_PRECISION         = 1e-8
ALT_MAKE_STOCHASTIC      = False
# learning rate
INITIAL_LEARNING_RATE    = 0.005
FINAL_LEARNING_RATE      = 0.0005
LEARNING_RATE_NUM_EPOCHS = 5
LEARNING_RATE_EPOCH      = MAX_ITERATIONS / (LEARNING_RATE_NUM_EPOCHS-1) # -1 because we want the number of epoch transitions
LEARNING_RATE_ADJUST     = (FINAL_LEARNING_RATE/INITIAL_LEARNING_RATE)**(-1.0/LEARNING_RATE_NUM_EPOCHS)

# Tensor type
DTYPE = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor

# Number of attempts
NUM_RAND_TRIALS          = 25
# Randomize initial tensor entries within +/-RAND_TRIAL_SCALE
RAND_TRIAL_SCALE         = 1

# We'll treat the transition matrices as a single rank-3 tensor
# indexed by (symbol,row,column)
#   dimension is the dimension of the state vector
#   symbol={0,1} (the transitions), to generalize
NSYM = 2
NDIM = 3
# ------------------------------------------------------------------------------

# Extract NumPy ndarray from a PyTorch tensor
def get_array(x):
	if torch.cuda.is_available():
		return x.cpu().detach().numpy()
	return x.detach().numpy()

# Contruct constrained stochastic tensor
def make_stochastic(T):
	if isinstance(T, np.ndarray):
		T = np.power(T, 2.0)
	else:
		T = torch.pow(T, 2.0)
	norms = T.sum(0).sum(1)
	for s in range(T.shape[0]):
		for i in range(T.shape[1]):
			T[s,i,:] /= norms[i]
	return T

# Compute probability given a proto-stoachstic tensor T and a sequence
def compute_probability(T, seq):
	T = make_stochastic(T)
	y = torch.zeros(T.shape[1]).type(DTYPE)
	y[0] = 1
	eta = torch.ones(T.shape[1]).type(DTYPE)
	# compute sequence
	for i in seq:
		y = y @ T[i]
	return -(y @ eta)

# Convert n to base b
def number_to_base(n, b, p=0):
	if n == 0:
		return [0]*(p - 1) + [0]
	digits = []
	while n:
		digits.append(int(n % b))
		n //= b
	return [0]*(p - len(digits)) + digits[::-1]

# ------------------------------------------------------------------------------
if not os.path.exists(PATH):
	os.makedirs(PATH)

# Optimize for all L
for L in range(3,10+1):
	# Note: originally, we optimized all the way to d=L-1 before we
	# developed the notion of deterministic complexity more formally.
	# We now know these are unnecessary, but it was still helpful to
	# understand state redundancies
	for NDIM in range(1,L): # go up to L-1, since L is trivial
		# Initialize starting points
		trials = []
		if NUM_RAND_TRIALS:
			trials += [torch.tensor(RAND_TRIAL_SCALE*(np.random.rand(NSYM,NDIM,NDIM)-0.5)*2.0,requires_grad=True).type(DTYPE)
					for i in range(NUM_RAND_TRIALS) ]
		NUM_TRIALS = len(trials)
		
		N_SEQS = NSYM**(L-1) # number of sequences (+1, since we'll use range())
		with open(os.path.join(PATH,"results_S%d_L%02d_D%02d.txt" % (NSYM, L, NDIM)), "w") as dat:
			for n in range(0,N_SEQS):
				
				seq = number_to_base(n,NSYM,L)
				seqstr = "".join([str(i) for i in seq])
				
				best_T = None
				best_p = 0
				Ts = []
				ps = []
				
				print("\n" + "-"*50)
				print("Optimizing... D=%d L=%d Sequence=%d/%d [%s]" % (NDIM, L, n, N_SEQS-1, seqstr))
				for tn, T0 in enumerate(trials):
					
					T = T0.clone().detach().requires_grad_()
					LR = INITIAL_LEARNING_RATE
					optimizer = torch.optim.Adam([T], LR)
					# This is the "pytorchian way" of doing epochs, but I couldn't get it to work
					# scheduler = ReduceLROnPlateau(optimizer, 'min')
					last_p = 2 # some initial large value, no need to be a valid one
					for it in range(MAX_ITERATIONS): # execute iterations
						res = compute_probability(T, seq)
						optimizer.zero_grad()
						res.backward()
						optimizer.step()
						p = -float(res)
						if abs(p - last_p) < TARGET_PRECISION:
							break
						last_p = p
						if ((it+1) % LEARNING_RATE_EPOCH) == 0: # for each epoch
							LR *= LEARNING_RATE_ADJUST # adjust the learning rate
							# update rate inside the model
							for param_group in optimizer.param_groups:
								param_group['lr'] = LR

					print("[Trial %03d/%03d] p=%0.10f after %d iterations" % (tn+1, NUM_TRIALS, p, it+1))

					if p > best_p:
						best_p = p
						best_T = make_stochastic(get_array(T))

					Ts.append(make_stochastic(get_array(T)))
					ps.append(p)
				
				print("Trials finished. Best p=%0.15f" % best_p)
				
				# Save model
				np.save(os.path.join(PATH,"T_S%d_L%02d_D%02d_%s.npy" % (NSYM, L, NDIM, seqstr)), best_T)
				dat.write("%d\t%s\t%0.15f\n" % (n, seqstr, best_p))
				dat.flush()


# EOF
