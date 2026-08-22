"""Ternary plaintext codec for the NTRU cryptosystem.

Encoding scheme (normative specification)
=========================================

Plaintext is an arbitrary byte string. The codec maps it deterministically
and reversibly onto a sequence of message polynomials in
R = Z[X]/(X^N - 1) with coefficients restricted to the balanced ternary
alphabet {-1, 0, +1}.

1. Length prefix.  The payload submitted to encoding is

       payload = len(data).to_bytes(4, "big") || data,

   where len(data) is the byte length of the original message. The 4-byte
   big-endian prefix bounds admissible messages to 2^32 - 1 bytes and makes
   decoding self-delimiting: trailing zero padding introduced by block
   alignment can be stripped unambiguously.

2. Base-3 digit expansion.  Every byte value b in [0, 256) is written as
   exactly six base-3 digits (little-endian, least significant digit first):

       b = sum_{d=0}^{5} digit_d * 3^d,   digit_d in {0, 1, 2}.

   Six digits are necessary and sufficient because 3^5 = 243 < 256 <= 729 = 3^6.

3. Digit-to-coefficient map.  Each ternary digit is mapped onto a ring
   coefficient via the balanced ternary embedding

       0 -> 0,   1 -> +1,   2 -> -1.

   This keeps the Euclidean representative of every message coefficient in
   {-1, 0, +1}, which minimizes the variance of f*m during decryption and
   therefore minimizes the decryption failure probability derived in the
   accompanying failure analysis.

4. Blocking.  A single message polynomial carries exactly BYTES_PER_BLOCK =
   N // 6 bytes, consuming 6 * (N // 6) <= N coefficient slots. Any residual
   high-degree coefficient slots of the polynomial (N mod 6 of them) are
   fixed to zero and are ignored by the decoder. The final partial block of
   the payload is zero-padded to a full block before polynomial mapping.

Decoding concatenates the recovered byte streams of all ciphertext blocks,
reads the 4-byte length prefix, validates it against the available byte
count, and returns exactly that many leading bytes.
"""

from __future__ import annotations

import struct
from typing import List, Sequence

import numpy as np

__all__ = ["TernaryCodec", "DIGITS_PER_BYTE"]

DIGITS_PER_BYTE = 6

_DIGIT_TO_COEFF = (0, 1, -1)
_COEFF_TO_DIGIT = {0: 0, 1: 1, -1: 2}

_LENGTH_PREFIX_BYTES = 4


class TernaryCodec:
    """Stateless encoder/decoder between byte strings and ternary polynomials."""

    def __init__(self, n: int) -> None:
        """Construct a codec for polynomials of dimension ``n``.

        Parameters
        ----------
        n:
            Ring dimension N of R = Z[X]/(X^N - 1).

        Raises
        ------
        ValueError
            If ``n`` is too small to hold even one encoded byte.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if n // DIGITS_PER_BYTE < 1:
            raise ValueError(
                "Ring dimension N=%d is too small: at least %d coefficients "
                "are required to encode one byte." % (n, DIGITS_PER_BYTE)
            )
        self._n = int(n)
        self.bytes_per_block = self._n // DIGITS_PER_BYTE

    @property
    def n(self) -> int:
        """Ring dimension served by this codec."""
        return self._n

    def encode(self, data: bytes) -> List[np.ndarray]:
        """Encode a byte string into a list of ternary message polynomials.

        Parameters
        ----------
        data:
            Arbitrary plaintext bytes.

        Returns
        -------
        list of numpy.ndarray
            One length-N integer array per block, each with coefficients in
            {-1, 0, +1}. The list always contains at least one polynomial,
            even for empty input.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be a bytes-like object.")
        data = bytes(data)
        if len(data) > 0xFFFFFFFF:
            raise ValueError(
                "Message exceeds the 2^32 - 1 byte limit imposed by the "
                "length prefix."
            )

        payload = struct.pack(">I", len(data)) + data
        bpb = self.bytes_per_block
        num_blocks = max(1, (len(payload) + bpb - 1) // bpb)
        padded = payload + b"\x00" * (num_blocks * bpb - len(payload))

        blocks: List[np.ndarray] = []
        for blk in range(num_blocks):
            coeffs = np.zeros(self._n, dtype=np.int64)
            offset = blk * bpb
            for j in range(bpb):
                byte = padded[offset + j]
                base = DIGITS_PER_BYTE * j
                for d in range(DIGITS_PER_BYTE):
                    coeffs[base + d] = _DIGIT_TO_COEFF[byte % 3]
                    byte //= 3
            blocks.append(coeffs)
        return blocks

    def decode(self, polynomials: Sequence[np.ndarray]) -> bytes:
        """Decode a sequence of ternary message polynomials back to bytes.

        Parameters
        ----------
        polynomials:
            Decrypted message polynomials with coefficients in {-1, 0, +1},
            as produced by :meth:`encode` and recovered by NTRU decryption.

        Returns
        -------
        bytes
            The original plaintext.

        Raises
        ------
        ValueError
            If any coefficient lies outside {-1, 0, +1, 2}, if no blocks are
            supplied, or if the embedded length prefix is inconsistent with
            the recovered byte stream.

        Notes
        -----
        Both canonical representations of R_p elements are accepted:
        balanced form {-1, 0, +1} (as produced by :meth:`encode`) and
        least-residue form {0, 1, 2} (as produced by ``NTRUEngine.decrypt``,
        which returns coefficients in [0, p)). The digit is recovered as
        ``coeff mod 3`` in either case.
        """
        if len(polynomials) == 0:
            raise ValueError("At least one message polynomial is required.")

        raw = bytearray()
        bpb = self.bytes_per_block
        for index, poly in enumerate(polynomials):
            arr = np.asarray(poly, dtype=np.int64)
            if arr.shape != (self._n,):
                raise ValueError(
                    "Message polynomial %d has shape %s; expected (%d,)."
                    % (index, arr.shape, self._n)
                )
            for j in range(bpb):
                value = 0
                power = 1
                for d in range(DIGITS_PER_BYTE):
                    coeff = int(arr[DIGITS_PER_BYTE * j + d])
                    if coeff not in (-1, 0, 1, 2):
                        raise ValueError(
                            "Coefficient %d of message polynomial %d has "
                            "value %d outside the ternary alphabet "
                            "{-1, 0, +1, 2}."
                            % (DIGITS_PER_BYTE * j + d, index, coeff)
                        )
                    digit = coeff % 3
                    value += digit * power
                    power *= 3
                raw.append(value)

        if len(raw) < _LENGTH_PREFIX_BYTES:
            raise ValueError("Recovered stream shorter than the length prefix.")
        declared_length = struct.unpack(">I", bytes(raw[:_LENGTH_PREFIX_BYTES]))[0]
        available = len(raw) - _LENGTH_PREFIX_BYTES
        if declared_length > available:
            raise ValueError(
                "Declared message length %d exceeds recovered payload size %d; "
                "decryption produced corrupt output." % (declared_length, available)
            )
        return bytes(raw[_LENGTH_PREFIX_BYTES:_LENGTH_PREFIX_BYTES + declared_length])