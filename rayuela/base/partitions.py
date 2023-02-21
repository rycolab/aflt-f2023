import random
import numpy as np
from collections import defaultdict as dd
from itertools import product

class PartitionRefinement:

	def __init__(self, f, Q):
	
		self.f = f
		self.Q = Q

		# compute the pre-image of f
		self.finv = dd(lambda : set([]))
		for n in Q: self.finv[self.f[n]].add(n)

	def stable(self, P):

		# definition of stable
		D = {}
		for n, B in enumerate(P):
			for q in B:
				D[q] = n

		for B in P:
			for p in B:
				for q in B:
					if D[self.f[p]] != D[self.f[q]]:
						return False
		return True

	def split(self, S, P):
		""" runs in O(|P|) time if Python is clever """
		return frozenset(P&S), frozenset(P-S)

	def naive(self, P):

		stack = list(P)

		while stack: # empties in O(|Q|) steps			
			S = stack.pop()
			R = set([]) # new refinement
			
			# compute subset of the pre-image in O(|Q|) time
			Sinv = set([]).union(*[self.finv[x] for x in S])

			for B in P: # entire loop runs in O(|Q|) time
				X, Y = self.split(Sinv, B) # runs in O(|B|) time

				if len(X) > 0 and len(Y) > 0:
					# X, Y are now part of the refinement
					R.add(X)
					R.add(Y)

					# X, Y become future splitters
					stack.append(X)
					stack.append(Y)
				else:
					# B remains part of the refinement
					R.add(B)
			P = R
			
		assert self.stable(P)
		return frozenset(P)

	def hopcroft(self, P):

		stack = list(P)
		while stack: # empties in O(log|Q|) steps			
			S = stack.pop()
			R = set([]) # new refinement
			
			# compute subset of the pre-image in O(|Q|) time
			Sinv = set([]).union(*[self.finv[x] for x in S])

			for B in P: # entire loop runs in O(|Q|) time
				X, Y = self.split(Sinv, B) # runs in O(|B|) time

				if len(X) > 0 and len(Y) > 0:
					# X, Y are now part of the refinement
					R.add(X)
					R.add(Y)

					if B in stack:	
						stack.remove(B)
						stack.append(X)
						stack.append(Y)
					else:
						if len(X) < len(Y):
							stack.append(X)
						else:
							stack.append(Y)

				else:
					# B remains part of the refinement
					R.add(B)
			P = R
			
		assert self.stable(P)
		return frozenset(P)

	def hopcroft_fast(self, P):

		P = list(map(set, P))
		N = len(P)
		stack = list(zip(P, range(len(P))))

		inblock = { b : (B, idx) for B, idx in stack for b in B }

		while stack: # empties in O(log |Q|) steps			
			(S, idx) = stack.pop()

			# computes subset of the pre-image
			# O(|Sinv|) time
			Sinv = set([]).union(*[self.finv[x] for x in S])

			# O(|Sinv|) time
			lst = [(inblock[s]) + (s,) for s in Sinv]

			# O(|Sinv|) time
			count = dd(lambda : 0)
			for _, idx, _ in lst:
				count[idx] += 1

			# excludes the case where a block B is a subset of Sinv
			# O(|Sinv|) time
			covered = set([ idx for B, idx, _ in lst if len(B) == count[idx]])

			# O(|Sinv|) time
			Rs = dd(lambda : set([]))
			for (B, idx, s) in lst:
				if idx in covered:
					continue
				B.remove(s)
				Rs[idx].add(s)

			# O(|Sinv|) time
			for idx, R in Rs.items():				
				for u in R:
					inblock[u] = (R, N)
				stack.append((R, N))
				P.append(R)
				N += 1

		P = frozenset(map(frozenset, P))
		return P

	def moore(self, P):

		# make all pairs of elements in O(Q²) time
		pairs = frozenset({ frozenset([p, q]) for p, q in product(self.Q, repeat=2) if p != q})

		# data structures
		lst = dd(set)
		diff = set([])

		# base case
		for B1, B2 in product(P, P):
			if B1 == B2:
				continue
			for p, q in product(B1, B2):
				diff.add(frozenset([p, q]))
		
		# main loop
		for pq in pairs:
			p, q = pq

			if frozenset([self.f[p], self.f[q]]) in diff:
				diff.add(pq)
			
				# recursively unmark
				stack = [pq]
				while stack:
					x = stack.pop()
					diff.add(x)
					for y in lst[x]:
						if y not in diff:
							stack.append(y)

			else:
				if self.f[p] != self.f[q]:
					lst[frozenset([self.f[p], self.f[q]])].add(pq)


		# create a partition of out pairwise differences
		P = {}
		for q in self.Q:
			P[q] = set([q])

		for pq in pairs - diff:
			p, q = pq
			new = P[p].union(P[q]).union([p, q])

			for p in new:
				P[p] = new

		final = set()
		for k, v in P.items():
			final.add(frozenset(v))

		return frozenset(final)


if __name__ == "__main__":
	#random.seed(0)
	#np.random.seed(0)
	for _ in range(100):
		N = random.randint(60, 100)
		L = random.randint(45, 55)

		# create random total function
		f = random.choices(range(N), k=N)	

		R = PartitionRefinement(f, frozenset(range(N)))

		# create random partition
		I = [0] + sorted(np.random.choice(range(2, N), L - 1, replace=False)) + [N]
		T = np.random.permutation(range(N))
		# P = { frozenset(P[:i]), frozenset(P[i:]) }
		P = set()
		for l in range(1, len(I)):
			P.add(frozenset(T[I[l - 1]: I[l]]))

		assert R.naive(P) == R.hopcroft(P)
		assert R.moore(P) == R.hopcroft(P)
		#print("Hopcroft", R.hopcroft_fast(P))
		#print("Moore", R.moore(P))
		assert R.moore(P) == R.hopcroft_fast(P)

	print('OK')