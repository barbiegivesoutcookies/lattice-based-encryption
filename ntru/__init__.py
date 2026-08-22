"""NTRU lattice-based cryptosystem library.

This package provides a from-scratch, dependency-light implementation of the
NTRU public-key encryption scheme over the polynomial quotient ring
R = Z[X]/(X^N - 1), together with a pure-Python textbook RSA baseline used
exclusively for benchmarking comparisons.

Security notice: this code is an academic reference implementation. It is
NOT audited, NOT constant-time, and MUST NOT be deployed to protect real
secrets.
"""

from ntru.engine import (
    NTRUEngine,
    NTRUInversionError,
    NTRUParameterError,
    PrivateKey,
    PublicKey,
)
from ntru.encoding import TernaryCodec

__all__ = [
    "NTRUEngine",
    "NTRUInversionError",
    "NTRUParameterError",
    "PrivateKey",
    "PublicKey",
    "TernaryCodec",
]