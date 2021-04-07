# Example implementation of DCPatterns in Python

import os

# Convert n to base b
def number_to_base(n, b, p=0):
	if n == 0:
		return [0]*(p - 1) + [0]
	digits = []
	while n:
		digits.append(int(n % b))
		n //= b
	return [0]*(p - len(digits)) + digits[::-1]

def DCPatterns(seq):
	L = len(seq)
	patts = []
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
				patts.append((seq[0:l1],seq[l1:l1+l2]))
	return dc, patts

def DCPatternStrings(patterns):
	pattstr = []
	for p in patterns:
		pattstr.append("%s(%s)" % ("".join(map(str,p[0])), "".join(map(str,p[1]))))
	return pattstr

L = 10
print("Seq\tDC\t#Patts\tPatterns")
for n in range(2**(L-1)):
	seq = number_to_base(n, 2, L)
	dc, patts = DCPatterns(seq)
	print("".join(map(str,seq)), "\t", dc, "\t", len(patts), "\t", ", ".join(DCPatternStrings(patts)))



