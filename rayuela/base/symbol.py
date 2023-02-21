from typing import Union, List, Set


class Sym:
    def __init__(self, sym):
        self.sym = sym

    def __str__(self):
        return str(self.sym)

    def __repr__(self):
        return str(self.sym)

    def __hash__(self):
        return hash(self.sym)

    def __eq__(self, other):
        return isinstance(other, Sym) and self.sym == other.sym

    def __invert__(self):
        return self


Alphabet = Set[Sym]


ε = Sym("ε")
ε_1 = Sym("ε_1")
ε_2 = Sym("ε_2")

φ = Sym("φ")
ρ = Sym("ρ")
σ = Sym("σ")

dummy = Sym("dummy")

BOS = Sym("<BOS>")
EOS = Sym("<EOS>")


def to_sym(s: str) -> Sym:
    """Converts a single character string to a symbol (Sym).

    Args:
        s (str): The input string

    Returns:
        Sym: Sym-ed version of the input string.
    """
    if isinstance(s, Sym):
        return s
    else:
        return Sym(s)


def to_alphabet(Sigma: Union[List[str], Set[str]]) -> Alphabet:
    """Converts a list input strings to a set of symbols.

    Args:
        Sigma (Union[List[str], Set[str]]): The input alphabet of non-symbol strings.

    Returns:
        Set[Sym]: The set of symbols.
    """
    return {to_sym(s) for s in Sigma}
