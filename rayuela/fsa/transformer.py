from typing import Tuple, Dict
from collections import defaultdict as dd
from itertools import chain, product
from frozendict import frozendict
from rayuela.base.semiring import Semiring

from rayuela.base.symbol import Sym, ε, φ, dummy
from rayuela.base.partitions import PartitionRefinement
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State, MinimizeState, PowerState
from rayuela.fsa.pathsum import Pathsum

class Transformer:

    @staticmethod
    def _powerarcs(fsa, Q):
        """This helper method group outgoing arcs for determinization."""

        symbol2arcs, unnormalized_residuals = dd(set), fsa.R.chart()

        for q, old_residual in Q.residuals.items():
            for a, p, w in fsa.arcs(q):
                symbol2arcs[a].add(p)
                unnormalized_residuals[(a, p)] += old_residual * w

        for a, ps in symbol2arcs.items():
            normalizer = sum(
                [unnormalized_residuals[(a, p)] for p in ps], start=fsa.R.zero
            )
            # this does not assume commutivity
            residuals = {p: ~normalizer * unnormalized_residuals[(a, p)] for p in ps}
            # this is an alternative formulation
            # residuals = {p: unnormalized_residuals[(a, p)] / normalizer for p in ps}

            yield a, PowerState(residuals), normalizer

    @staticmethod
    def push(fsa):
        from rayuela.fsa.pathsum import Strategy

        W = Pathsum(fsa).backward(Strategy.LEHMANN)
        pfsa = Transformer._push(fsa, W)
        assert pfsa.pushed  # sanity check
        return pfsa

    @staticmethod
    def _push(fsa, V):
        """
        Mohri (2001)'s weight pushing algorithm. See Eqs 1, 2, 3.
        Link: https://www.isca-speech.org/archive_v0/archive_papers/eurospeech_2001/e01_1603.pdf.
        """

        pfsa = fsa.spawn()
        for i in fsa.Q:
            pfsa.set_I(i, fsa.λ[i] * V[i])
            print(i, fsa.λ[i] * V[i])
            pfsa.set_F(i, ~V[i] * fsa.ρ[i])
            for a, j, w in fsa.arcs(i):
                pfsa.add_arc(i, a, j, ~V[i] * w * V[j])

        return pfsa

    # TODO (Anej): Rename this method after grading...
    @staticmethod
    def _eps_partition(fsa, partition_symbol: Sym = ε) -> Tuple[FSA, FSA]:
        """Partition FSA into two (one with arcs of the partition symbol and one with all others)

        Args:
            fsa (FSA): The input FSA
            partition_symbol (Sym, optional): The symbol based on which to partition the input FSA

        Returns:
            Tuple[FSA, FSA]: The FSA with non-partition symbol arcs
                             and the FSA with only the partition symbol arcs
        """

        E = fsa.spawn()
        N = fsa.spawn(keep_init=True, keep_final=True)

        for q in fsa.Q:
            E.add_state(q)
            N.add_state(q)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                if a == partition_symbol:
                    E.add_arc(i, a, j, w)
                else:
                    N.add_arc(i, a, j, w)

        return N, E

    @staticmethod
    def epsremoval(fsa):

        # note that N keeps same initial and final weights
        N, E = Transformer._eps_partition(fsa)
        W = Pathsum(E).lehmann(zero=False)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i, no_eps=True):
                for k in fsa.Q:
                    N.add_arc(i, a, k, w * W[j, k])

        # additional initial states
        for i, j in product(fsa.Q, repeat=2):
            N.add_I(j, fsa.λ[i] * W[i, j])

        return N

    @staticmethod
    def twins(fsa):
        """
        Alluzen and Mohri's (2003) algorithm for testing whether a WFSA has the twins property.
        Time complexity: O(Q² + E²)
        """
        from rayuela.fsa.scc import SCC

        F = fsa.intersect(fsa.invert())

        scc = SCC(F)
        for c in scc.scc():
            Fscc = scc.to_fsa(c)
            # print(list(Fscc.Q))
            # test the cycle identity on every SCC
            if not Transformer.cycle_identity(Fscc):
                return False

        return True