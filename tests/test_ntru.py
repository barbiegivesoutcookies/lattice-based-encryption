"""Automated correctness and security test suite for the NTRU implementation.

Run from the project root with:

    python -m unittest discover -s tests -v

All randomness is drawn from explicitly seeded ``random.Random`` instances so
that every assertion in this file is fully reproducible.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from ntru.engine import (
    NTRUEngine,
    NTRUInversionError,
    NTRUParameterError,
)
from ntru.encoding import TernaryCodec


class TestNTRURoundTripCorrectness(unittest.TestCase):
    """decrypt(encrypt(m)) == m over seeded randomized key/message pairs."""

    def test_round_trip_correctness(self) -> None:
        """200 seeded iterations at the N=167 tier: full pipeline round trip.

        Each iteration generates a fresh key pair, a fresh random plaintext
        of random length, encodes it through the ternary codec, encrypts
        every block with fresh blinding polynomials, decrypts, decodes, and
        asserts exact byte-level equality with the original message.
        """
        rng = random.Random(42)
        engine = NTRUEngine(n=167, p=3, q=512, df=45, dg=20, dr=20)
        codec = TernaryCodec(engine.n)

        for iteration in range(200):
            public_key, private_key = engine.generate_keypair(rng)
            message_length = rng.randint(1, 120)
            message = bytes(rng.randrange(256) for _ in range(message_length))

            message_blocks = codec.encode(message)
            ciphertext_blocks = [
                engine.encrypt(m, public_key, rng) for m in message_blocks
            ]
            recovered_blocks = [
                engine.decrypt(c, private_key) for c in ciphertext_blocks
            ]
            recovered_message = codec.decode(recovered_blocks)

            self.assertEqual(
                recovered_message,
                message,
                msg="Round-trip mismatch at iteration %d." % iteration,
            )

    def test_round_trip_production_parameters(self) -> None:
        """Seeded round trips at the full academic parameter set (N=509)."""
        rng = random.Random(2024)
        engine = NTRUEngine()  # N=509, p=3, q=2048, d_f=216, d_g=d_r=72
        codec = TernaryCodec(engine.n)

        for iteration in range(25):
            public_key, private_key = engine.generate_keypair(rng)
            message_length = rng.randint(1, 250)
            message = bytes(rng.randrange(256) for _ in range(message_length))

            message_blocks = codec.encode(message)
            ciphertext_blocks = [
                engine.encrypt(m, public_key, rng) for m in message_blocks
            ]
            recovered_blocks = [
                engine.decrypt(c, private_key) for c in ciphertext_blocks
            ]

            self.assertEqual(codec.decode(recovered_blocks), message)

    def test_inverse_congruence_invariants(self) -> None:
        """f * f_p^{-1} = 1 (mod p) and f * f_q^{-1} = 1 (mod q) on fresh keys."""
        rng = random.Random(777)
        engine = NTRUEngine(n=167, p=3, q=512, df=45, dg=20, dr=20)
        identity = np.zeros(engine.n, dtype=np.int64)
        identity[0] = 1  # multiplicative identity of R is the constant 1
        for _ in range(10):
            public_key, private_key = engine.generate_keypair(rng)
            check_p = engine.polymul_mod(private_key.f, private_key.f_p_inv, engine.p)
            check_q = engine.polymul_mod(private_key.f, private_key.f_q_inv, engine.q)
            np.testing.assert_array_equal(check_p % engine.p, identity)
            np.testing.assert_array_equal(check_q % engine.q, identity)


class TestEmpiricalDecryptionFailureRate(unittest.TestCase):
    """Empirical failure rates versus the theoretical bound of Section 1."""

    def test_empirical_decryption_failure_rate_boundary_parameters(self) -> None:
        """Large seeded batch at the boundary of the norm bounds.

        The stress parameter set (N=53, p=3, q=64, d_f=17, d_g=d_r=8) places
        sigma_C^2 close enough to (q/2)^2 that coefficient wrap-around occurs
        at an observable rate. The empirical rate must not exceed the
        heuristic bound Pr[failure] <= 2*N*exp(-q^2 / (8*sigma_C^2)) derived
        in the proposal's Decryption Failure Analysis subsection.
        """
        rng = random.Random(987654321)
        engine = NTRUEngine(n=53, p=3, q=64, df=17, dg=8, dr=8)
        trials = 4000
        failures = 0

        for _ in range(trials):
            public_key, private_key = engine.generate_keypair(rng)
            m = engine.random_message_poly(rng)
            e = engine.encrypt(m, public_key, rng)
            recovered = engine.decrypt(e, private_key)
            if not np.array_equal(recovered, m % engine.p):
                failures += 1

        empirical_rate = failures / trials
        theoretical_bound = engine.decryption_failure_bound()
        variance = engine.decryption_variance()

        print(
            "[boundary stress set] N=%d p=%d q=%d d_f=%d d_g=%d d_r=%d "
            "sigma_C^2=%.2f" % (
                engine.n, engine.p, engine.q, engine.df, engine.dg, engine.dr,
                variance,
            )
        )
        print(
            "[boundary stress set] trials=%d failures=%d "
            "empirical_rate=%.6f theoretical_bound=%.6f"
            % (trials, failures, empirical_rate, theoretical_bound)
        )

        # The heuristic bound must dominate the observed rate.
        self.assertLessEqual(
            empirical_rate,
            theoretical_bound,
            msg=(
                "Empirical decryption failure rate %.6f exceeds the "
                "theoretical bound %.6f." % (empirical_rate, theoretical_bound)
            ),
        )

    def test_production_parameter_batch_no_failures(self) -> None:
        """Batch at the academic parameter set: zero failures expected.

        For (N=509, p=3, q=2048, d_f=216, d_g=d_r=72) the heuristic bound is
        below 10^-300, so any single failure out of this batch would falsify
        the parameter analysis.
        """
        rng = random.Random(2718281828)
        engine = NTRUEngine()
        trials = 300
        failures = 0

        for _ in range(trials):
            public_key, private_key = engine.generate_keypair(rng)
            m = engine.random_message_poly(rng)
            e = engine.encrypt(m, public_key, rng)
            recovered = engine.decrypt(e, private_key)
            if not np.array_equal(recovered, m % engine.p):
                failures += 1

        bound = engine.decryption_failure_bound()
        print(
            "[production set] trials=%d failures=%d theoretical_bound=%.3e"
            % (trials, failures, bound)
        )
        self.assertEqual(failures, 0)
        self.assertLess(bound, 1e-100)


class TestParameterValidation(unittest.TestCase):
    """Invalid parameter sets must be rejected by the constructor."""

    def test_rejects_non_prime_n(self) -> None:
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=512, p=3, q=2048, df=216, dg=72, dr=72)

    def test_rejects_n_equal_one(self) -> None:
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=1, p=3, q=16, df=1, dg=1, dr=1)

    def test_rejects_gcd_p_q_not_one(self) -> None:
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=167, p=4, q=2048, df=45, dg=20, dr=20)

    def test_rejects_q_not_power_of_two(self) -> None:
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=167, p=3, q=1000, df=45, dg=20, dr=20)

    def test_rejects_oversized_hamming_weights(self) -> None:
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=53, p=3, q=64, df=54, dg=8, dr=8)
        with self.assertRaises(NTRUParameterError):
            NTRUEngine(n=53, p=3, q=64, df=17, dg=27, dr=8)

    def test_valid_parameter_set_accepted(self) -> None:
        engine = NTRUEngine(n=107, p=3, q=128, df=29, dg=10, dr=10)
        self.assertEqual(engine.n, 107)


class TestTernaryCodec(unittest.TestCase):
    """Direct codec unit checks complementing the end-to-end round trips."""

    def test_codec_round_trip_all_byte_values(self) -> None:
        rng = random.Random(314159)
        codec = TernaryCodec(167)
        payload = bytes(range(256)) + bytes(rng.randrange(256) for _ in range(500))
        blocks = codec.encode(payload)
        for block in blocks:
            self.assertTrue(np.all(np.isin(block, (-1, 0, 1))))
        self.assertEqual(codec.decode(blocks), payload)

    def test_codec_rejects_corrupt_coefficients(self) -> None:
        codec = TernaryCodec(167)
        blocks = codec.encode(b"attack at dawn")
        corrupted = [block.copy() for block in blocks]
        corrupted[0][7] = 5  # outside the ternary alphabet
        with self.assertRaises(ValueError):
            codec.decode(corrupted)

    def test_empty_message_round_trip(self) -> None:
        codec = TernaryCodec(509)
        blocks = codec.encode(b"")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(codec.decode(blocks), b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)