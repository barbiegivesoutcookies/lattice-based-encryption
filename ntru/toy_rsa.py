"""Textbook (raw, unpadded) RSA implemented from scratch.

Purpose and scope
=================
This module exists for exactly one reason: to serve as the classical
baseline in the benchmarking chapter. RSA is not the subject under test;
it is the control group against which NTRU key generation, encryption and
decryption latencies are compared across equivalent security tiers.

SECURITY WARNING: this is textbook RSA. It uses no randomized padding
scheme (no OAEP), so it is deterministic, malleable, and vulnerable to a
long list of classical attacks (low-exponent broadcast, chosen-ciphertext,
Wiener/Coppersmith on small parameters). It must never be used outside of
this benchmarking context. Its inclusion here is also a reminder that RSA's
hardness assumption (integer factorization) is broken in polynomial time by
Shor's algorithm on a sufficiently large fault-tolerant quantum computer,
whereas no such quantum algorithm is known for the lattice problems
underlying NTRU — a conjecture, not a theorem.
"""

from __future__ import annotations

import random
import struct
from typing import List, Tuple

__all__ = ["ToyRSA"]

_SMALL_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251,
)

_LENGTH_PREFIX_BYTES = 4


def _is_probable_prime(n: int, rng: random.Random, rounds: int = 16) -> bool:
    """Miller-Rabin probabilistic primality test with ``rounds`` random bases.

    Error probability per invocation is at most 4^(-rounds) for odd
    composite inputs; combined with trial division by small primes this is
    overwhelmingly reliable for benchmark-scale key generation.
    """
    if n < 2:
        return False
    for prime in _SMALL_PRIMES:
        if n % prime == 0:
            return n == prime
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int, rng: random.Random) -> int:
    """Generate a random probable prime with exactly ``bits`` bits."""
    if bits < 2:
        raise ValueError("Prime bit length must be at least 2.")
    while True:
        candidate = rng.getrandbits(bits)
        candidate |= (1 << (bits - 1)) | 1  # force top bit and oddness
        if _is_probable_prime(candidate, rng):
            return candidate


class ToyRSA:
    """Raw RSA key pair with byte-string encrypt/decrypt helpers."""

    def __init__(self, key_bits: int, rng: random.Random) -> None:
        """Generate an RSA key pair whose modulus has approximately ``key_bits`` bits.

        Parameters
        ----------
        key_bits:
            Target modulus size in bits (e.g. 512, 1024, 2048).
        rng:
            Seeded randomness source for reproducible benchmarks.
        """
        if key_bits < 8:
            raise ValueError("key_bits must be at least 8.")
        half = key_bits // 2
        p = _generate_prime(half, rng)
        q = _generate_prime(key_bits - half, rng)
        while q == p:
            q = _generate_prime(key_bits - half, rng)
        self.p = p
        self.q = q
        self.n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        if e >= phi or _gcd(e, phi) != 1:
            e = 3
            while _gcd(e, phi) != 1:
                e += 2
        self.e = e
        self.d = pow(e, -1, phi)
        self.key_bits = self.n.bit_length()
        # A block of this many bytes, interpreted as an integer, is strictly
        # less than the modulus: 256**block_size <= 2**(n_bits - 1) < n.
        self.block_size = (self.key_bits - 1) // 8

    # ------------------------------------------------------------------
    # Integer primitives
    # ------------------------------------------------------------------

    def encrypt_int(self, m: int) -> int:
        """Raw RSA encryption c = m^e mod n."""
        if not 0 <= m < self.n:
            raise ValueError("Plaintext integer out of range [0, n).")
        return pow(m, self.e, self.n)

    def decrypt_int(self, c: int) -> int:
        """Raw RSA decryption m = c^d mod n."""
        if not 0 <= c < self.n:
            raise ValueError("Ciphertext integer out of range [0, n).")
        return pow(c, self.d, self.n)

    # ------------------------------------------------------------------
    # Byte-string interface
    # ------------------------------------------------------------------

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt a byte string block-by-block with raw RSA.

        The payload carries a 4-byte big-endian length prefix; each block of
        ``block_size`` bytes is encrypted independently as an integer.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be a bytes-like object.")
        data = bytes(data)
        payload = struct.pack(">I", len(data)) + data
        blocks_needed = max(1, (len(payload) + self.block_size - 1) // self.block_size)
        padded = payload + b"\x00" * (blocks_needed * self.block_size - len(payload))
        ct_bytes_per_block = (self.key_bits + 7) // 8
        output = bytearray()
        for i in range(blocks_needed):
            chunk = padded[i * self.block_size:(i + 1) * self.block_size]
            c = self.encrypt_int(int.from_bytes(chunk, "big"))
            output.extend(c.to_bytes(ct_bytes_per_block, "big"))
        return bytes(output)

    def decrypt_bytes(self, ciphertext: bytes) -> bytes:
        """Decrypt output produced by :meth:`encrypt_bytes`."""
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise TypeError("ciphertext must be a bytes-like object.")
        ciphertext = bytes(ciphertext)
        ct_bytes_per_block = (self.key_bits + 7) // 8
        if len(ciphertext) == 0 or len(ciphertext) % ct_bytes_per_block != 0:
            raise ValueError(
                "Ciphertext length %d is not a positive multiple of %d."
                % (len(ciphertext), ct_bytes_per_block)
            )
        plain = bytearray()
        for i in range(0, len(ciphertext), ct_bytes_per_block):
            chunk = ciphertext[i:i + ct_bytes_per_block]
            c = int.from_bytes(chunk, "big")
            m = self.decrypt_int(c)
            plain.extend(m.to_bytes(self.block_size, "big"))
        if len(plain) < _LENGTH_PREFIX_BYTES:
            raise ValueError("Recovered stream shorter than the length prefix.")
        declared_length = struct.unpack(">I", bytes(plain[:_LENGTH_PREFIX_BYTES]))[0]
        available = len(plain) - _LENGTH_PREFIX_BYTES
        if declared_length > available:
            raise ValueError("Declared length exceeds recovered payload size.")
        return bytes(plain[_LENGTH_PREFIX_BYTES:_LENGTH_PREFIX_BYTES + declared_length])


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return abs(a)