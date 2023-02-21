from typing import Optional, Type

import numpy as np

from fractions import Fraction
from collections import defaultdict as dd
from frozendict import frozendict

from math import exp, log


# base code from https://github.com/timvieira/hypergraphs/blob/master/hypergraphs/semirings/boolean.py
class Semiring:

    zero: "Semiring"
    one: "Semiring"
    idempotent = False

    def __init__(self, score):
        self.score = score

    @classmethod
    def zeros(cls, N, M):
        import numpy as np

        return np.full((N, M), cls.zero)

    @classmethod
    def chart(cls, default=None):
        if default is None:
            default = cls.zero
        return dd(lambda: default)

    @classmethod
    def diag(cls, N):
        W = cls.zeros(N, N)
        for n in range(N):
            W[n, n] = cls.one

        return W

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return self.score == other.score

    def __hash__(self):
        return hash(self.score)


class Derivation(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return Derivation.one

    def __add__(self, other):
        return Derivation(frozenset(self.score.union(other.score)))

    def __mul__(self, other):
        # TODO: add special cases
        return Derivation(frozenset([x + y for x in self.score for y in other.score]))

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"Derivation({self.score})"

    def __hash__(self):
        return hash(self.score)


Derivation.zero = Derivation(frozenset([]))
Derivation.one = Derivation(frozenset([tuple()]))
Derivation.idempotent = False


class KBest(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def __add__(self, other):
        return KBest(self.score.union(other.score))

    def __mul__(self, other):
        # TODO: add special cases
        return KBest(set([x + y for x in self.score for y in other.score]))

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"KBest({self.score})"

    def __hash__(self):
        return hash(self.score)


KBest.zero = Derivation(set([]))
KBest.one = Derivation(set([tuple()]))
KBest.idempotent = False


class Free(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return "(" + self.score + ")^*"

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Free(self.score + " + " + other.score)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Free(self.score + other.score)

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"Free({self.score})"

    def __hash__(self):
        return hash(self.score)


Free.zero = Free("∞")
Free.one = Free("")
Free.idempotent = False


class Count(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return self.one

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Count(self.score + other.score)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Count(self.score * other.score)

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"{self.score}"

    def __hash__(self):
        return hash(self.score)

    def __float__(self):
        return float(self.score)


Count.zero = Count(0)
Count.one = Count(1)
Count.idempotent = False


class Entropy(Semiring):
    def __init__(self, x, y):
        super().__init__((x, y))

    def star(self):
        tmp = 1.0 / (1.0 - self.score[0])
        return Entropy(tmp, tmp * tmp * self.score[1])

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Entropy(self.score[0] + other.score[0], self.score[1] + other.score[1])

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Entropy(
            self.score[0] * other.score[0],
            self.score[0] * other.score[1] + self.score[1] * other.score[0],
        )

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"Entropy({self.score})"

    def __hash__(self):
        return hash(self.score)


Entropy.zero = Entropy(0.0, 0.0)
Entropy.one = Entropy(1.0, 0.0)
Entropy.idempotent = False


def cky_semiring_builder(G, R):

    one = "1"

    class CKY(Semiring):
        def __init__(self, score):
            super().__init__(frozendict(score))

        def star(self):
            return CKY.one

        def __add__(self, other):
            if other is self.zero:
                return self
            if self is self.zero:
                return other

            result = dd(lambda: self.R.zero)
            for k, v in self.score.items():
                result[k] += v
            for k, v in other.score.items():
                result[k] += v

            return CKY(frozendict(result))

        def __mul__(self, other):
            if other is self.one:
                return self
            if self is self.one:
                return other
            if other is self.zero:
                return self.zero
            if self is self.zero:
                return self.zero

            result = dd(lambda: self.R.zero)

            # special handling of "1" symbol
            if one in self.score:
                for nt, v in other.score.items():
                    result[nt] += v
            if one in other.score:
                for nt, v in self.score.items():
                    result[nt] += v

            # Cartesian product subject to grammar constraint
            for p, w in self.G.binary:
                if p.body[0] in self.score and p.body[1] in other.score:
                    result[p.head] += self.score[p.body[0]] * other.score[p.body[1]] * w

            return CKY(frozendict(result))

        def __eq__(self, other):
            return self.score == other.score

        def __repr__(self):
            return f"{self.score}"

        def __hash__(self):
            return hash(self.score)

    CKY.G = G
    CKY.R = R
    CKY.zero = CKY(dict())
    CKY.one = CKY({one: CKY.R.one})
    CKY.idempotent = False
    CKY.cancellative = False

    return CKY


class String(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return String.one

    def __add__(self, other):
        from rayuela.base.misc import lcp

        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return String(lcp(self.score, other.score))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return String(self.score + other.score)

    def __truediv__(self, other):
        from rayuela.base.misc import lcp

        prefix = lcp(self.score, other.score)
        return String(self.score[len(prefix) :])

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"{self.score}"

    def __hash__(self):
        return hash(self.score)


# unique "infinity" string
String.zero = String("∞")
# empty string
String.one = String("")
String.idempotent = False
String.cancellative = False


class Segment:
    def __init__(self, segment, inverse=False):
        self.segment = segment
        self.inverse = inverse

    def __invert__(self):
        return Segment(self.segment, inverse=not self.inverse)

    def __eq__(self, other):
        return self.segment == other.segment and self.inverse == other.inverse

    def __repr__(self):
        return f"{self.segment, self.inverse}"

    def __str__(self):
        return f"{self.segment}"

    def __hash__(self):
        return hash((self.segment, self.inverse))


class SegmentationGroup(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return SegmentationGroup.one

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other

        if len(self.score) < len(other.score):
            return self
        return other

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero

        def simplify(arg1, arg2):
            changed = True
            while changed:
                changed = False
                if len(arg1) == 0 or len(arg2) == 0:
                    break

                if ~arg1[-1] == arg2[0]:
                    arg1 = arg1[:-1]
                    arg2 = arg2[1:]
                    changed = True

            return arg1 + arg2

        # make this better
        simplified = tuple(
            [x for x in simplify(self.score, other.score) if x.segment != ""]
        )
        return SegmentationGroup(simplified)

    def __invert__(self):
        return SegmentationGroup(tuple(reversed([~x for x in self.score])))

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f"{'|'.join(map(str, self.score))}"

    def __hash__(self):
        return hash(self.score)


# unique "infinity" string
SegmentationGroup.zero = SegmentationGroup(None)
# empty string
# TODO: inverse
SegmentationGroup.one = SegmentationGroup(())
SegmentationGroup.idempotent = False
SegmentationGroup.cancellative = False


class Boolean(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return Boolean.one

    def __add__(self, other):
        return Boolean(self.score or other.score)

    def __mul__(self, other):
        if other.score is self.one:
            return self.score
        if self.score is self.one:
            return other.score
        if other.score is self.zero:
            return self.zero
        if self.score is self.zero:
            return self.zero
        return Boolean(other.score and self.score)

    # TODO: is this correct?
    def __invert__(self):
        return Boolean.one

    def __truediv__(self, other):
        return Boolean.one

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"{self.score}"

    def __str__(self):
        return str(self.score)

    def __hash__(self):
        return hash(self.score)


Boolean.zero = Boolean(False)
Boolean.one = Boolean(True)
Boolean.idempotent = True
# TODO: check
Boolean.cancellative = True


class MaxPlus(Semiring):
    def __init__(self, score):
        super().__init__(score)

    def star(self):
        return self.one

    def __float__(self):
        return float(self.score)

    def __add__(self, other):
        return MaxPlus(max(self.score, other.score))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return MaxPlus(self.score + other.score)

    def __invert__(self):
        return MaxPlus(-self.score)

    def __truediv__(self, other):
        return MaxPlus(self.score - other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"MaxPlus({self.score})"


MaxPlus.zero = MaxPlus(float("-inf"))
MaxPlus.one = MaxPlus(0.0)
MaxPlus.idempotent = True
MaxPlus.superior = True
MaxPlus.cancellative = True


class Tropical(Semiring):
    def __init__(self, score):
        self.score = score

    def star(self):
        return self.one

    def __float__(self):
        return float(self.score)

    def __int__(self):
        return int(self.score)

    def __add__(self, other):
        return Tropical(min(self.score, other.score))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Tropical(self.score + other.score)

    def __invert__(self):
        return Tropical(-self.score)

    def __truediv__(self, other):
        return Tropical(self.score - other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"Tropical({self.score})"

    def __str__(self):
        return str(self.score)


Tropical.zero = Tropical(float("inf"))
Tropical.one = Tropical(0.0)
Tropical.idempotent = True
Tropical.superior = True
Tropical.cancellative = True


class Rational(Semiring):
    def __init__(self, score):
        self.score = Fraction(score)

    def star(self):
        return Rational(Fraction("1") / (Fraction("1") - self.score))

    def __float__(self):
        return float(self.score)

    def __add__(self, other):
        return Rational(self.score + other.score)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Rational(self.score * other.score)

    def __invert__(self):
        return Rational(1 / self.score)

    def __truediv__(self, other):
        return Rational(self.score / other.score)

    def __eq__(self, other):
        return np.allclose(float(self.score), float(other.score))

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return f'Real({self.score})'
        return f"{self.score}"

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.score)


Rational.zero = Rational(Fraction("0"))
Rational.one = Rational(Fraction("1"))
Rational.idempotent = False
Rational.cancellative = True


class Real(Semiring):
    def __init__(self, score):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.score = score

    def star(self):
        return Real(1.0 / (1.0 - self.score))

    def __float__(self):
        return float(self.score)

    def __add__(self, other):
        return Real(self.score + other.score)

    def __sub__(self, other):
        return Real(self.score - other.score)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Real(self.score * other.score)

    def __invert__(self):
        return Real(1.0 / self.score)

    def __pow__(self, other):
        return Real(self.score**other)

    def __truediv__(self, other):
        return Real(self.score / other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return f'Real({self.score})'
        return f"{round(self.score, 15)}"

    def __eq__(self, other):
        # return float(self.score) == float(other.score)
        return np.allclose(float(self.score), float(other.score), atol=1e-6)

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.score)


Real.zero = Real(0.0)
Real.one = Real(1.0)
Real.idempotent = False
Real.cancellative = True


class Log(Semiring):
    def __init__(self, score):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.score = score

    def star(self):
        return Log(-log(1 / exp(self.score) - 1) - self.score)

    def __float__(self):
        return float(self.score)

    def __add__(self, other):
        # stolen from https://github.com/timvieira/crf/blob/master/crf/basecrf.py
        if self.score > other.score:
            return Log(self.score + log(exp(other.score - self.score) + 1))
            # return Log(self.score + log(sum(exp(other.score-self.score)).sum()))
        return Log(other.score + log(exp(self.score - other.score + 1)))

    def __mul__(self, other):
        return Log(self.score + other.score)

    def __repr__(self):
        # return f'Real({self.score})'
        return f"{round(self.score, 15)}"

    def __eq__(self, other):
        # return float(self.score) == float(other.score)
        return np.allclose(float(self.score), float(other.score), atol=1e-3)

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.score)


Log.zero = Log(-float("inf"))
Log.one = Log(0.0)
Log.idempotent = False
Log.cancellative = True


class Integer(Semiring):
    def __init__(self, score):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.score = score

    def __float__(self):
        return float(self.score)

    def __add__(self, other):
        return Integer(self.score + other.score)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Integer(self.score * other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"Integer({self.score})"

    def __eq__(self, other):
        return float(self.score) == float(other.score)

    def __hash__(self):
        return hash(self.score)


Integer.zero = Integer(0)
Integer.one = Integer(1)
Integer.idempotent = False
Integer.cancellative = True


def vector_semiring_builder(semiring, N):
    class VectorSemiring(Semiring):
        def __init__(self, x):
            super().__init__(x)

        def star(self):
            raise NotImplemented

        def __add__(self, other):
            return VectorSemiring(self.score + other.score)

        def __mul__(self, other):
            return VectorSemiring(self.score * other.score)

        def __eq__(self, other):
            return self.score == other.score

        def __repr__(self):
            return f"Vector({self.score})"

        def __hash__(self):
            return hash(self.score)

    VectorSemiring.zero = VectorSemiring(np.full(N, semiring.zero))
    VectorSemiring.one = VectorSemiring(np.full(N, semiring.one))
    VectorSemiring.idempotent = semiring.idempotent

    return VectorSemiring


class ProductSemiring(Semiring):
    def __init__(self, x, y):
        super().__init__((x, y))

    def star(self):
        raise NotImplemented

    def __add__(self, other):
        w1, w2 = self.score[0], other.score[0]
        v1, v2 = self.score[1], other.score[1]
        return ProductSemiring(w1 + w2, v1 + v2)

    def __mul__(self, other):
        w1, w2 = self.score[0], other.score[0]
        v1, v2 = self.score[1], other.score[1]
        return ProductSemiring(w1 * w2, v1 * v2)

    def __truediv__(self, other):
        w1, w2 = self.score[0], other.score[0]
        v1, v2 = self.score[1], other.score[1]
        return ProductSemiring(w1 / w2, v1 / v2)

    def __invert__(self):
        return ProductSemiring(~self.score[0], ~self.score[1])

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        if isinstance(self.score[0], String):
            # the imporant special case of encoding transducers
            if len(self.score[0].score) > 0:
                return f"{self.score[0]} / {self.score[1]}"
            else:
                return f"{self.score[1]}"
        return f"〈{self.score[0]}, {self.score[1]}〉"

    def __hash__(self):
        return hash(self.score)


def product_semiring_builder(semiring1, semiring2):

    ProductSemiring.zero = ProductSemiring(semiring1.zero, semiring2.zero)
    ProductSemiring.one = ProductSemiring(semiring1.one, semiring2.one)
    ProductSemiring.idempotent = semiring1.idempotent and semiring2.idempotent

    return ProductSemiring


def expectation_semiring_builder(semiring1, semiring2):
    class ExpectationSemiring(Semiring):
        def __init__(self, x, y):
            super().__init__((x, y))

        def star(self):
            w, v = self.score[0], self.score[1]
            return ExpectationSemiring(w.star(), w.star() * w.star() * v)

        def __add__(self, other):
            w1, w2 = self.score[0], other.score[0]
            v1, v2 = self.score[1], other.score[1]
            return ExpectationSemiring(w1 + w2, v1 + v2)

        def __mul__(self, other):
            w1, w2 = self.score[0], other.score[0]
            v1, v2 = self.score[1], other.score[1]
            return ExpectationSemiring(w1 * w2, w1 * v2 + w2 * v1)

        def __eq__(self, other):
            return self.score == other.score

        def __repr__(self):
            return f"Expect({self.score})"

        def __hash__(self):
            return hash(self.score)

    ExpectationSemiring.zero = ExpectationSemiring(semiring1.zero, semiring2.zero)
    ExpectationSemiring.one = ExpectationSemiring(semiring1.one, semiring2.zero)
    ExpectationSemiring.idempotent = semiring1.idempotent and semiring2.idempotent

    return ExpectationSemiring


class NullableSemiring(Semiring):
    null: "NullableSemiring"

    def __init__(self, x: Optional[Semiring] = None):
        if x is not None:
            # NullableSemiring can only be used as a wrapper around another semiring,
            # not itself
            assert not isinstance(x, NullableSemiring)
            super().__init__(x)
            self.R_base = type(x)
            self.isnull = False
        else:
            self.isnull = True

    def __add__(self, other):
        if self == NullableSemiring.null:
            return other
        elif other == NullableSemiring.null:
            return self
        else:
            return NullableSemiring(self.score + other.score)

    def __sub__(self, other):
        if self == NullableSemiring.null:
            return -other
        elif other == NullableSemiring.null:
            return self
        else:
            return NullableSemiring(self.score - other.score)

    def __mul__(self, other):
        if NullableSemiring.null in [self, other]:
            return NullableSemiring.null
        else:
            return NullableSemiring(self.score * other.score)

    def __eq__(self, other):
        if self.isnull and other.isnull:
            return True
        elif self.isnull or other.isnull:
            return False
        else:
            return self.score == other.score

    def star(self):
        if self != NullableSemiring.null:
            return self.score.star()
        else:
            raise TypeError

    def __float__(self):
        if self != NullableSemiring.null:
            return float(self.score)
        else:
            raise TypeError

    def __invert__(self):
        if self != NullableSemiring.null:
            return ~self.score
        else:
            raise TypeError

    def __truediv__(self, other):
        if self != NullableSemiring.null:
            return self.score / other.score
        else:
            raise TypeError

    def __lt__(self, other):
        if self != NullableSemiring.null:
            return self.score < other.score
        else:
            raise TypeError

    def __repr__(self):
        if self != NullableSemiring.null:
            return self.score.__repr__()
        else:
            return "null"

    def __str__(self):
        if self != NullableSemiring.null:
            return self.score.__str__()
        else:
            return "null"

    def __hash__(self):
        if self != NullableSemiring.null:
            return hash(self.score)
        else:
            raise TypeError


NullableSemiring.null = NullableSemiring()


def nullable_semiring_builder(R: Type[Semiring]) -> Type[NullableSemiring]:

    NullableSemiring.zero = NullableSemiring(R.zero)
    NullableSemiring.one = NullableSemiring(R.one)

    return NullableSemiring


def conditionalpoisson_semiring_builder(K):
    class ConditionalPoisson(Semiring):
        def __init__(self, x):
            super().__init__(x)

        def star(self):
            raise NotImplemented

        def __add__(self, other):
            return ConditionalPoisson(np.convolve(self.score, other.score)[:K])

        def __mul__(self, other):
            return ConditionalPoisson(self.score * other.score)

        def __eq__(self, other):
            return (
                isinstance(other, ConditionalPoisson)
                and (self.score == other.score).all()
            )

        def __repr__(self):
            return str(self.score)

        def __hash__(self):
            return hash(self.score)

    tmp = np.zeros((K))
    tmp[0] = 1
    ConditionalPoisson.zero = ConditionalPoisson(tmp)
    ConditionalPoisson.one = ConditionalPoisson(np.ones((K)))

    ConditionalPoisson.idempotent = False
    ConditionalPoisson.cancellative = False

    return ConditionalPoisson
