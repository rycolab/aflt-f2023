from collections import deque

from rayuela.base.semiring import Boolean, Real
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.fsa import FSA


class SCC:
    def __init__(self, fsa, single_I=True):
        self.fsa = fsa.single_I() if single_I else fsa.copy()
        self.R = self.fsa.R

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self) -> "list[frozenset]":
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """
        # Assignment 3: Question 4
        raise NotImplementedError