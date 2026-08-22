"""Core NTRU engine over the polynomial quotient ring R = Z[X]/(X^N - 1).

Representation conventions
=========================
Every ring element is a NumPy ``int64`` array of length exactly N holding
coefficients in little-endian degree order: index i stores the coefficient
of X^i. Reduction of an element of R modulo p or modulo q is coefficient-wise
and always yields representatives in [0, modulus). Centered lifting maps
representatives in [0, q) to the symmetric window [-q/2, q/2).

Algebraic pipeline
==================
* Multiplication in R is cyclic convolution (O(N^2) schoolbook algorithm).
  The Number Theoretic Transform is NOT applicable for q = 2^k: the unit
  group (Z/2^kZ)^x has order 2^(k-1) and therefore contains no element of
  odd prime order N, so no primitive N-th root of unity exists in Z/qZ.
* Inversion modulo the prime p uses the Extended Euclidean Algorithm in the
  genuine finite field F_p[X], computing u, v with u*f + v*(X^N - 1) = c.
* Inversion modulo q = 2^k uses Hensel lifting from the mod-p inverse. The
  Extended Euclidean Algorithm cannot be applied directly over Z/qZ because
  Z/qZ is not a field when q is composite (zero divisors exist), so Bezout
  coefficients need not exist even for units.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "NTRUEngine",
    "NTRUInversionError",
    "NTRUParameterError",
    "PrivateKey",
    "PublicKey",
]


class NTRUParameterError(ValueError):
    """Raised when an NTRU parameter set violates required structural constraints."""


class NTRUInversionError(RuntimeError):
    """Raised when a private polynomial admits no inverse in R_p or R_q."""


def _is_prime(n: int) -> bool:
    """Deterministic trial-division primality test.

    Adequate for NTRU dimension selection, where candidate values of N are
    small primes (a few hundred to a few thousand).
    """
    if not isinstance(n, int) or n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _gcd_int(a: int, b: int) -> int:
    """Greatest common divisor of two integers (Euclid's algorithm)."""
    while b:
        a, b = b, a % b
    return abs(a)


@dataclass(frozen=True)
class PublicKey:
    """NTRU public key.

    Mathematical content: the ring element h = p * f^{-1} * g in R_q,
    where f, g are the (secret) ternary polynomials of the key holder.
    """

    n: int
    p: int
    q: int
    h: np.ndarray

    def to_dict(self) -> dict:
        """JSON-serializable representation of the public key."""
        return {
            "n": int(self.n),
            "p": int(self.p),
            "q": int(self.q),
            "h": [int(c) for c in self.h],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PublicKey":
        """Rebuild a PublicKey from its dictionary form, validating structure."""
        try:
            n = int(payload["n"])
            p = int(payload["p"])
            q = int(payload["q"])
            h_list = payload["h"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed public key payload.") from exc
        if not isinstance(h_list, (list, tuple)):
            raise ValueError("Public key component 'h' must be a list of integers.")
        if len(h_list) != n:
            raise ValueError(
                "Public key 'h' has length %d; expected N=%d." % (len(h_list), n)
            )
        h = np.array([int(c) for c in h_list], dtype=np.int64)
        if np.any(h < 0) or np.any(h >= q):
            raise ValueError("Public key coefficients must lie in [0, q).")
        return cls(n=n, p=p, q=q, h=h)


@dataclass(frozen=True)
class PrivateKey:
    """NTRU private key.

    Mathematical content: the ternary polynomial f together with its
    inverses F_p = f^{-1} in R_p and F_q = f^{-1} in R_q, plus the paired
    public element h for convenience of transport.
    """

    n: int
    p: int
    q: int
    f: np.ndarray
    f_p_inv: np.ndarray
    f_q_inv: np.ndarray
    h: np.ndarray

    def to_dict(self) -> dict:
        """JSON-serializable representation of the private key."""
        return {
            "n": int(self.n),
            "p": int(self.p),
            "q": int(self.q),
            "f": [int(c) for c in self.f],
            "f_p_inv": [int(c) for c in self.f_p_inv],
            "f_q_inv": [int(c) for c in self.f_q_inv],
            "h": [int(c) for c in self.h],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PrivateKey":
        """Rebuild a PrivateKey from its dictionary form, validating structure."""
        try:
            n = int(payload["n"])
            p = int(payload["p"])
            q = int(payload["q"])
            components = {
                name: payload[name] for name in ("f", "f_p_inv", "f_q_inv", "h")
            }
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed private key payload.") from exc
        arrays = {}
        for name, value in components.items():
            if not isinstance(value, (list, tuple)) or len(value) != n:
                raise ValueError(
                    "Private key component '%s' must be a list of length N=%d."
                    % (name, n)
                )
            arr = np.array([int(c) for c in value], dtype=np.int64)
            arrays[name] = arr
        if np.any(arrays["f_q_inv"] < 0) or np.any(arrays["f_q_inv"] >= q):
            raise ValueError("f_q_inv coefficients must lie in [0, q).")
        if np.any(arrays["h"] < 0) or np.any(arrays["h"] >= q):
            raise ValueError("h coefficients must lie in [0, q).")
        if np.any(arrays["f_p_inv"] < 0) or np.any(arrays["f_p_inv"] >= p):
            raise ValueError("f_p_inv coefficients must lie in [0, p).")
        return cls(n=n, p=p, q=q, **arrays)


class NTRUEngine:
    """Encapsulates the algebra of R = Z[X]/(X^N - 1) modulo p and modulo q.

    Default parameter set (academic reference tier):

        N = 509 (prime), p = 3, q = 2048 = 2^11,
        d_f = 216 (+1 coefficients, with d_f - 1 = 215 coefficients equal -1),
        d_g = d_r = 72.

    These parameters follow the structural conventions of the EESS/NTRU
    parameter families; they are intended for study and benchmarking, not as
    a certified production profile.
    """

    def __init__(
        self,
        n: int = 509,
        p: int = 3,
        q: int = 2048,
        df: int = 216,
        dg: int = 72,
        dr: int = 72,
        max_keygen_attempts: int = 100,
    ) -> None:
        """Construct an engine and validate its parameter set.

        Raises
        ------
        NTRUParameterError
            If any structural constraint on (N, p, q, d_f, d_g, d_r) fails.
        """
        self.n = int(n)
        self.p = int(p)
        self.q = int(q)
        self.df = int(df)
        self.dg = int(dg)
        self.dr = int(dr)
        self.max_keygen_attempts = int(max_keygen_attempts)
        self.validate_parameters()

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        """Enforce all structural constraints on the parameter set.

        Constraints enforced here:

        1. N must be prime. Primality of N guarantees that X^N - 1 factors
           over F_p as (X - 1) * Phi_N(X), which keeps the invertibility
           analysis of f tractable.
        2. gcd(p, q) = 1. Required by the Chinese-remainder/Hensel argument
           that reconstructs f's inverse modulo pq from inverses modulo p
           and modulo q.
        3. q must be a power of two, q = 2^k with k >= 3. This is the regime
           targeted by the Hensel-lifting inversion schedule implemented in
           :meth:`inv_mod_q`.
        4. Ternary densities must satisfy 1 <= d_f <= N, 1 <= d_g <= N // 2,
           1 <= d_r <= N // 2, keeping f, g, r sparse relative to N.
        """
        if not _is_prime(self.n):
            raise NTRUParameterError(
                "N = %r is not prime; NTRU requires a prime ring dimension." % (self.n,)
            )
        if _gcd_int(self.p, self.q) != 1:
            raise NTRUParameterError(
                "gcd(p, q) = %d must equal 1; got p = %d, q = %d."
                % (_gcd_int(self.p, self.q), self.p, self.q)
            )
        if self.q < 8 or (self.q & (self.q - 1)) != 0:
            raise NTRUParameterError(
                "q = %r must be a power of two (q = 2^k, k >= 3) for the "
                "Hensel-lifting inversion schedule." % (self.q,)
            )
        if self.p < 2:
            raise NTRUParameterError("p = %r must be at least 2." % (self.p,))
        if not (1 <= self.df <= self.n):
            raise NTRUParameterError(
                "d_f = %r outside admissible range [1, N]." % (self.df,)
            )
        if not (1 <= self.dg <= self.n // 2):
            raise NTRUParameterError(
                "d_g = %r outside admissible range [1, N//2]." % (self.dg,)
            )
        if not (1 <= self.dr <= self.n // 2):
            raise NTRUParameterError(
                "d_r = %r outside admissible range [1, N//2]." % (self.dr,)
            )

    # ------------------------------------------------------------------
    # Ring arithmetic in R = Z[X]/(X^N - 1)
    # ------------------------------------------------------------------

    def cyclic_convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cyclic convolution of two ring elements over the integers.

        Computes c in Z[X]/(X^N - 1) with

            c_k = sum_{i=0}^{N-1} a_i * b_{(k - i) mod N},

        i.e. multiplication followed by reduction modulo X^N - 1 via the
        wrap-around identification X^{N} = 1. Complexity is O(N^2); the NTT
        acceleration is unavailable because q = 2^k admits no primitive
        N-th root of unity (see module docstring).

        Parameters
        ----------
        a, b:
            Length-N integer coefficient arrays.

        Returns
        -------
        numpy.ndarray
            Exact integer convolution product, length N, dtype int64.
        """
        a = self._coerce_ring_element(a, "a")
        b = self._coerce_ring_element(b, "b")
        accumulator = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            scalar = int(a[i])
            if scalar != 0:
                accumulator += scalar * np.roll(b, i)
        return accumulator

    def polymul_mod(self, a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
        """Multiply two ring elements and reduce coefficient-wise mod ``modulus``."""
        product = self.cyclic_convolve(a, b)
        return product % int(modulus)

    def polyadd_mod(self, a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
        """Add two ring elements and reduce coefficient-wise mod ``modulus``."""
        a = self._coerce_ring_element(a, "a")
        b = self._coerce_ring_element(b, "b")
        return ((a + b) % int(modulus)).astype(np.int64)

    def center_lift(self, a: np.ndarray, modulus: int) -> np.ndarray:
        """Centered lift: map residues in [0, m) to the window [-m/2, m/2).

        For odd m the image is [-(m-1)/2, (m-1)/2]; for even m it is
        [-m/2, m/2 - 1]. Centered lifting is the operation that makes NTRU
        decryption exact whenever no coefficient of f*e wraps around q.
        """
        arr = np.asarray(a, dtype=np.int64)
        m = int(modulus)
        return ((arr + m // 2) % m) - m // 2

    def sample_ternary(
        self, count_plus_one: int, count_minus_one: int, rng: random.Random
    ) -> np.ndarray:
        """Sample a uniformly random ternary polynomial with prescribed Hamming weight.

        Exactly ``count_plus_one`` coefficients are drawn as +1, exactly
        ``count_minus_one`` as -1, and the remaining N - (d1 + d2) as 0.
        Sampling uses Fisher-Yates style index selection without replacement.
        """
        total = count_plus_one + count_minus_one
        if total > self.n:
            raise ValueError(
                "Requested Hamming weight %d exceeds ring dimension %d."
                % (total, self.n)
            )
        indices = rng.sample(range(self.n), total)
        poly = np.zeros(self.n, dtype=np.int64)
        for idx in indices[:count_plus_one]:
            poly[idx] = 1
        for idx in indices[count_plus_one:]:
            poly[idx] = -1
        return poly

    def random_message_poly(self, rng: random.Random) -> np.ndarray:
        """Sample a dense message polynomial with i.i.d. coefficients in {-1, 0, +1}.

        Each coefficient is uniform on the balanced ternary alphabet,
        giving Var(m_i) = 2/3 — the message model assumed by
        :meth:`decryption_failure_bound`.
        """
        return np.array(
            [rng.randrange(-1, 2) for _ in range(self.n)], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Polynomial arithmetic over the finite field F_p[X]
    # ------------------------------------------------------------------

    @staticmethod
    def _poly_trim(a: np.ndarray) -> np.ndarray:
        """Strip highest-degree zero coefficients (no-op for the zero polynomial)."""
        end = a.size
        while end > 0 and a[end - 1] == 0:
            end -= 1
        return a[:end]

    @staticmethod
    def _poly_mul_plain(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
        """Ordinary (non-cyclic) polynomial multiplication over F_p."""
        if a.size == 0 or b.size == 0:
            return np.zeros(0, dtype=np.int64)
        out = np.zeros(a.size + b.size - 1, dtype=np.int64)
        for i in range(a.size):
            coeff = int(a[i])
            if coeff != 0:
                out[i:i + b.size] += coeff * b
        return out % p

    @staticmethod
    def _poly_sub(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
        """Shape-safe subtraction a - b over F_p.

        Operands may have different lengths (and may be empty, representing
        the zero polynomial). NumPy's broadcasting rules would silently
        collapse ``a - empty`` to an empty array, which would corrupt the
        Bezout coefficient sequences inside the Extended Euclidean Algorithm;
        this helper pads to the maximum length before subtracting.
        """
        size = max(a.size, b.size)
        out = np.zeros(size, dtype=np.int64)
        out[:a.size] += a
        out[:b.size] -= b
        return out % p

    def _poly_divmod(self, a: np.ndarray, b: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """Polynomial division a = q*b + r over the field F_p.

        Requires b nonzero; every nonzero leading coefficient is invertible
        because F_p is a field. Returns (quotient, remainder).
        """
        dividend = self._poly_trim(np.asarray(a, dtype=np.int64) % p)
        divisor = self._poly_trim(np.asarray(b, dtype=np.int64) % p)
        if divisor.size == 0:
            raise ZeroDivisionError("Polynomial division by zero in F_p[X].")
        inv_lead = pow(int(divisor[-1]), -1, p)
        remainder = dividend.copy()
        if remainder.size < divisor.size:
            return np.zeros(0, dtype=np.int64), remainder
        quotient = np.zeros(remainder.size - divisor.size + 1, dtype=np.int64)
        for pos in range(remainder.size - divisor.size, -1, -1):
            lead = int(remainder[pos + divisor.size - 1])
            if lead != 0:
                factor = (lead * inv_lead) % p
                quotient[pos] = factor
                remainder[pos:pos + divisor.size] = (
                    remainder[pos:pos + divisor.size] - factor * divisor
                ) % p
        return quotient, self._poly_trim(remainder[:divisor.size - 1])

    def _poly_xgcd(
        self, a: np.ndarray, b: np.ndarray, p: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extended Euclidean Algorithm in F_p[X].

        Returns (g, u, v) with u*a + v*b = g, where g = gcd(a, b) normalized
        monic. Valid because F_p is a field; this is precisely why the same
        routine cannot be reused modulo composite q.
        """
        r_prev = self._poly_trim(np.asarray(a, dtype=np.int64) % p)
        r_cur = self._poly_trim(np.asarray(b, dtype=np.int64) % p)
        s_prev = np.ones(1, dtype=np.int64)
        s_cur = np.zeros(1, dtype=np.int64)
        t_prev = np.zeros(1, dtype=np.int64)
        t_cur = np.ones(1, dtype=np.int64)

        while r_cur.size > 0:
            quot, rem = self._poly_divmod(r_prev, r_cur, p)
            r_prev, r_cur = r_cur, rem
            s_prev, s_cur = (
                s_cur,
                self._poly_sub(s_prev, self._poly_mul_plain(quot, s_cur, p), p),
            )
            t_prev, t_cur = (
                t_cur,
                self._poly_sub(t_prev, self._poly_mul_plain(quot, t_cur, p), p),
            )

        g = r_prev
        if g.size == 0:
            return g, s_prev, t_prev
        inv_lead = pow(int(g[-1]), -1, p)
        g = (g * inv_lead) % p
        u = (s_prev * inv_lead) % p
        v = (t_prev * inv_lead) % p
        return g, u, v

    def _fold_reduce_mod_xn_minus_1(self, coeffs: np.ndarray, p: int) -> np.ndarray:
        """Reduce an arbitrary coefficient vector modulo (X^N - 1) over F_p."""
        out = np.zeros(self.n, dtype=np.int64)
        for i in range(coeffs.size):
            out[i % self.n] = (out[i % self.n] + coeffs[i]) % p
        return out

    def _inv_mod_field(self, f: np.ndarray, prime: int) -> np.ndarray:
        """Inverse of f in R_prime = F_prime[X]/(X^N - 1) for prime ``prime``.

        Method: Extended Euclidean Algorithm over the genuine finite field
        F_prime. We compute u, v in F_prime[X] with

            u(X) * f(X) + v(X) * (X^N - 1) = c(X).

        If c is a nonzero constant, reducing the identity modulo (X^N - 1)
        gives u * f = c in R_prime, hence f^{-1} = c^{-1} * u. If deg(c) > 0
        then gcd(f, X^N - 1) is nontrivial and f is a zero divisor.

        Raises
        ------
        NTRUInversionError
            If f is not invertible in R_prime.
        """
        f_arr = self._coerce_ring_element(f, "f") % prime
        x_n_minus_1 = np.zeros(self.n + 1, dtype=np.int64)
        x_n_minus_1[0] = (-1) % prime
        x_n_minus_1[self.n] = 1
        g, u, _v = self._poly_xgcd(f_arr, x_n_minus_1, prime)
        if g.size != 1:
            raise NTRUInversionError(
                "f is not invertible modulo %d: gcd(f, X^N - 1) is nontrivial "
                "in F_%d[X]." % (prime, prime)
            )
        inv_const = pow(int(g[0]), -1, prime)
        scaled = (u * inv_const) % prime
        if scaled.size <= self.n:
            result = np.zeros(self.n, dtype=np.int64)
            result[:scaled.size] = scaled
            return result
        return self._fold_reduce_mod_xn_minus_1(scaled, prime)

    def inv_mod_p(self, f: np.ndarray) -> np.ndarray:
        """Inverse of f in R_p = F_p[X]/(X^N - 1).

        Delegates to :meth:`_inv_mod_field`: the Extended Euclidean Algorithm
        over the genuine finite field F_p (p prime). See that method for the
        mathematical justification.

        Raises
        ------
        NTRUInversionError
            If f is not invertible in R_p.
        """
        return self._inv_mod_field(f, self.p)

    def inv_mod_q(self, f: np.ndarray, f_p_inv: np.ndarray = None) -> np.ndarray:
        """Inverse of f in R_q = (Z/qZ)[X]/(X^N - 1) via Hensel lifting.

        Why not the Extended Euclidean Algorithm directly? Because
        q = 2^k is composite, Z/qZ is NOT a field: it contains zero divisors,
        so the division step of the Euclidean algorithm is undefined for
        non-invertible leading coefficients and Bezout-type identities need
        not exist. Instead we lift the known inverse modulo p:

        Hensel lifting step (doubling with binary correction). Suppose
        f * fi = Id + t*w for some integer polynomial w, where
        Id = (1, 0, ..., 0) is the multiplicative identity of R, i.e.
        f * fi = Id (mod t). Reducing modulo 2t and using the invariant,

            prod = (f * fi) mod 2t = Id + t*e,   e in {0, 1}^N,

        where the division by t is EXACT coefficientwise by the invariant.
        Seek fi' = fi + t*u with u in {0, 1}^N such that f*fi' = Id (mod 2t):

            f*fi' = f*fi + t*u*f = Id + t*(e + u*f)   (mod 2t),

        so we need e + u*f = 0 (mod 2), i.e. u = e * f^{-1} (mod 2), where
        f^{-1} in R_2 is precomputed once via the Extended Euclidean
        Algorithm over F_2. This restores the invariant modulo 2t.

        NOTE on why the naive squaring form fi' = fi*(2 - f*fi) mod 2t is
        NOT used here: squaring lifts validity from mod t to mod t^2, and
        congruence modulo t^2 implies congruence modulo 2t only when t is
        even. Starting at odd t = p, reducing intermediate results modulo
        2t destroys the invariant; the explicit correction step above is
        the mathematically sound doubling lift.

        Schedule. Starting from the exact inverse modulo t = p and applying
        the doubling step log2(q) times,

            t : p -> 2p -> 4p -> ... -> p*q,

        terminates at an inverse valid modulo p*q. Since q divides p*q,
        reducing the final representative modulo q yields the exact inverse
        of f in R_q. (The schedule reaches p*q exactly because q = 2^k.)

        Parameters
        ----------
        f:
            Ring element to invert.
        f_p_inv:
            Precomputed inverse of f in R_p; computed on demand if omitted.

        Returns
        -------
        numpy.ndarray
            f^{-1} in R_q with coefficients in [0, q).

        Raises
        ------
        NTRUInversionError
            If f is not invertible modulo p, or if the final verification
            f * f^{-1} = 1 (mod q) fails.
        """
        if f_p_inv is None:
            f_p_inv = self.inv_mod_p(f)
        else:
            f_p_inv = self._coerce_ring_element(f_p_inv, "f_p_inv")

        # Auxiliary inverse of f in R_2 for the binary Hensel correction.
        f_inv2 = self._inv_mod_field(f, 2)

        identity = np.zeros(self.n, dtype=np.int64)
        identity[0] = 1  # multiplicative identity of R is the constant 1

        target = self.p * self.q
        t = self.p
        fi = (f_p_inv % self.p).astype(np.int64)
        while t < target:
            prod = self.polymul_mod(f, fi, 2 * t)      # == Id + t*e (mod 2t)
            e = ((prod - identity) // t) % 2           # exact division by invariant
            u = self.polymul_mod(e, f_inv2, 2)         # u = e * f^{-1} (mod 2)
            fi = ((fi + t * u) % (2 * t)).astype(np.int64)
            t *= 2
        fi = fi % self.q

        check = self.polymul_mod(f, fi, self.q)
        if not np.array_equal(check, identity):
            raise NTRUInversionError(
                "Hensel lifting failed to produce a verified inverse modulo q."
            )
        return fi

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def generate_keypair(
        self, rng: random.Random
    ) -> Tuple[PublicKey, PrivateKey]:
        """Generate an NTRU key pair.

        Algorithm
        ---------
        1. Sample f from L(d_f, d_f - 1): d_f coefficients equal +1 and
           d_f - 1 equal -1. The asymmetric choice forces f(1) = 1, which
           guarantees that (X - 1) does not divide f over any F_p, removing
           one systematic obstruction to invertibility in R_p.
        2. Compute F_p = f^{-1} in R_p (Extended Euclidean Algorithm) and
           F_q = f^{-1} in R_q (Hensel lifting). If either fails, resample f.
        3. Sample g from L(d_g, d_g).
        4. Publish h = p * F_q * g mod q.

        Correctness invariant: for any ciphertext e = r*h + m (mod q),

            f * e = p*r*g + f*m   (mod q),

        which is the identity exploited by :meth:`decrypt`.

        Parameters
        ----------
        rng:
            Source of randomness (e.g. ``random.Random`` seeded for
            reproducibility, or ``random.SystemRandom`` for operational use).

        Returns
        -------
        (PublicKey, PrivateKey)

        Raises
        ------
        NTRUInversionError
            If no invertible f was found within ``max_keygen_attempts``.
        """
        for _attempt in range(self.max_keygen_attempts):
            f = self.sample_ternary(self.df, self.df - 1, rng)
            try:
                f_p_inv = self.inv_mod_p(f)
                f_q_inv = self.inv_mod_q(f, f_p_inv)
            except NTRUInversionError:
                continue
            g = self.sample_ternary(self.dg, self.dg, rng)
            h = self.polymul_mod(g, f_q_inv, self.q)
            h = (int(self.p) * h) % self.q
            public_key = PublicKey(n=self.n, p=self.p, q=self.q, h=h.copy())
            private_key = PrivateKey(
                n=self.n,
                p=self.p,
                q=self.q,
                f=f.copy(),
                f_p_inv=f_p_inv.copy(),
                f_q_inv=f_q_inv.copy(),
                h=h.copy(),
            )
            return public_key, private_key
        raise NTRUInversionError(
            "Key generation failed: no invertible f found within %d attempts."
            % self.max_keygen_attempts
        )

    # ------------------------------------------------------------------
    # Encryption / decryption
    # ------------------------------------------------------------------

    def encrypt(
        self, message: np.ndarray, public_key: PublicKey, rng: random.Random
    ) -> np.ndarray:
        """Encrypt one message polynomial.

        Computes e = r*h + m (mod q), where r is a fresh random blinding
        polynomial sampled from L(d_r, d_r). Fresh randomness per encryption
        is essential: reusing r across two messages leaks their difference
        under a known-ciphertext comparison.

        Parameters
        ----------
        message:
            Plaintext ring element with small coefficients (the codec in
            :mod:`ntru.encoding` produces coefficients in {-1, 0, +1}).
        public_key:
            Recipient's public key.
        rng:
            Randomness source for the blinding polynomial.

        Returns
        -------
        numpy.ndarray
            Ciphertext ring element with coefficients in [0, q).
        """
        m = self._coerce_ring_element(message, "message")
        if public_key.n != self.n or public_key.p != self.p or public_key.q != self.q:
            raise ValueError("Public key parameters do not match engine parameters.")
        r = self.sample_ternary(self.dr, self.dr, rng)
        rh = self.polymul_mod(r, public_key.h, self.q)
        return (rh + m) % self.q

    def decrypt(self, ciphertext: np.ndarray, private_key: PrivateKey) -> np.ndarray:
        """Decrypt one ciphertext polynomial.

        Pipeline
        --------
        1. a = f * e mod q. Expanding e = r*h + m with h = p*F_q*g gives
           a = p*r*g + f*m (mod q) over the integers.
        2. Centered-lift a into [-q/2, q/2). Whenever every coefficient of
           the integer polynomial p*r*g + f*m already lies in that window,
           the lift recovers it exactly (this is the probabilistic step;
           see :meth:`decryption_failure_bound`).
        3. Reduce mod p: the term p*r*g vanishes, leaving f*m mod p.
        4. Multiply by F_p: m = F_p * (f*m) = m (mod p).

        Returns
        -------
        numpy.ndarray
            Recovered message polynomial with coefficients in [0, p).

        Raises
        ------
        ValueError
            If the ciphertext shape is wrong or the key does not match.
        """
        e = self._coerce_ring_element(ciphertext, "ciphertext")
        if private_key.n != self.n or private_key.p != self.p or private_key.q != self.q:
            raise ValueError("Private key parameters do not match engine parameters.")
        a = self.polymul_mod(private_key.f, e, self.q)
        a_centered = self.center_lift(a, self.q)
        m = self.polymul_mod(private_key.f_p_inv, a_centered, self.p)
        return m

    # ------------------------------------------------------------------
    # Decryption failure analysis support
    # ------------------------------------------------------------------

    def decryption_variance(self) -> float:
        """Variance sigma_C^2 of a coefficient of C = f*m + p*r*g.

        Model: f, g, r are independent ternary polynomials with Hamming
        weights (2*d_f - 1), (2*d_g), (2*d_r); m has i.i.d. uniform
        {-1, 0, +1} coefficients (Var = 2/3). A convolution coefficient is a
        sum of N independent products of independent centered variables, so

            Var[(a*b)_k] = N * Var[a_i] * Var[b_i],

        with Var[f_i] = (2 d_f - 1)/N, Var[g_i] = 2 d_g / N,
        Var[r_i] = 2 d_r / N. Hence

            sigma_C^2 = N * (sigma_f^2 * sigma_m^2 + p^2 * sigma_r^2 * sigma_g^2).
        """
        var_f = (2 * self.df - 1) / self.n
        var_g = (2 * self.dg) / self.n
        var_r = (2 * self.dr) / self.n
        var_m = 2.0 / 3.0
        return self.n * (var_f * var_m + (self.p ** 2) * var_r * var_g)

    def decryption_failure_bound(self) -> float:
        """Heuristic upper bound on the per-message decryption failure probability.

        Under the standard Gaussian/independence heuristic used throughout
        the NTRU literature (cf. IEEE P1363.1 parameter-estimation practice),
        each coefficient of C = f*m + p*r*g is modeled as a centered random
        variable with variance sigma_C^2, giving the Chernoff-style tail

            Pr[|C_k| >= q/2]  <=  2 * exp(-q^2 / (8 * sigma_C^2)).

        A decryption failure occurs iff ANY of the N coefficients wraps
        around q, so the union bound yields

            Pr[failure]  <=  2*N*exp(-q^2 / (8*sigma_C^2)).

        IMPORTANT epistemic status: this is a heuristic estimate, not a
        proven theorem — the independence/Gaussian assumption on convolution
        coefficients is empirically well-supported but unproven. A fully
        rigorous bound via Cantelli's inequality (one-sided Chebyshev),

            Pr[|C_k| >= q/2] <= 4*sigma_C^2/q^2,

        is valid but far too loose to guide parameter selection (for the
        default parameter set it exceeds 0.6 after the union bound, while
        the heuristic bound is below 10^-300 and empirical rates are zero).

        Returns
        -------
        float
            min(1.0, 2*N*exp(-q^2 / (8*sigma_C^2))).
        """
        variance = self.decryption_variance()
        exponent = -(float(self.q) ** 2) / (8.0 * variance)
        return min(1.0, 2.0 * self.n * math.exp(exponent))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_ring_element(self, value, name: str) -> np.ndarray:
        """Validate and convert input to a length-N int64 coefficient array."""
        arr = np.asarray(value)
        if arr.dtype.kind not in "iu":
            raise TypeError(
                "%s must contain integer coefficients; got dtype %s."
                % (name, arr.dtype)
            )
        arr = arr.astype(np.int64)
        if arr.shape != (self.n,):
            raise ValueError(
                "%s must have exactly N=%d coefficients; got shape %s."
                % (name, self.n, arr.shape)
            )
        return arr