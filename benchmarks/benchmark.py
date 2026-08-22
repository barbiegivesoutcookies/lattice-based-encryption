"""Benchmarking suite: NTRU lattice cryptosystem versus textbook RSA.

Methodology
===========
* All randomness is derived from a single fixed seed (SEED) so that every
  reported number is reproducible bit-for-bit on the same interpreter build.
* Timing uses ``timeit.Timer.repeat``; each reported figure is the MINIMUM
  total time over REPEAT samples divided by NUMBER calls, i.e. the
  best-case per-operation latency. Minimum-of-repeats is the standard
  noise-robust estimator for microbenchmarks (cf. Python documentation,
  timeit module guidance).
* Workload: a fixed 64-byte message, encrypted and decrypted end-to-end
  (including codec/blocking overhead) in one timed operation.
* Tiers:
    NTRU : (N=107, q=512), (N=167, q=512), (N=509, q=2048)
    RSA  : 512-bit, 1024-bit, 2048-bit moduli.
* Outputs: a printed table, benchmarks/benchmark_results.csv, and a grouped
  bar chart with logarithmic latency axis at
  benchmarks/benchmark_results.png.

Run from the project root:

    python -m benchmarks.benchmark
"""

from __future__ import annotations

import csv
import os
import random
import sys
import timeit

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display server required
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntru.engine import NTRUEngine  # noqa: E402
from ntru.encoding import TernaryCodec  # noqa: E402
from ntru.toy_rsa import ToyRSA  # noqa: E402

SEED = 20260822
MESSAGE_BYTES = 64
REPEAT = 5


def _time_per_call(func, number: int, repeat: int = REPEAT) -> float:
    """Best-case seconds per call via timeit.repeat."""
    timer = timeit.Timer(func)
    samples = timer.repeat(repeat=repeat, number=number)
    return min(samples) / number


def benchmark_ntru_tier(
    n: int, q: int, df: int, dg: int, dr: int, rng: random.Random
) -> dict:
    """Benchmark keygen / encrypt / decrypt for one NTRU parameter tier."""
    engine = NTRUEngine(n=n, p=3, q=q, df=df, dg=dg, dr=dr)
    codec = TernaryCodec(n)
    public_key, private_key = engine.generate_keypair(rng)
    message = bytes(rng.randrange(256) for _ in range(MESSAGE_BYTES))
    blocks = codec.encode(message)

    def do_keygen() -> None:
        engine.generate_keypair(rng)

    def do_encrypt() -> None:
        for m in codec.encode(message):
            engine.encrypt(m, public_key, rng)

    def do_decrypt() -> None:
        ciphertexts = [engine.encrypt(m, public_key, rng) for m in blocks]
        recovered = [engine.decrypt(c, private_key) for c in ciphertexts]
        assert codec.decode(recovered) == message

    # Sanity check before timing.
    do_decrypt()

    return {
        "scheme": "NTRU",
        "tier": "N=%d,q=%d" % (n, q),
        "keygen_s": _time_per_call(do_keygen, number=3),
        "encrypt_s": _time_per_call(do_encrypt, number=10),
        "decrypt_s": _time_per_call(do_decrypt, number=10),
    }


def benchmark_rsa_tier(key_bits: int, rng: random.Random) -> dict:
    """Benchmark keygen / encrypt / decrypt for one RSA modulus size."""
    rsa = ToyRSA(key_bits, rng)
    message = bytes(rng.randrange(256) for _ in range(MESSAGE_BYTES))
    ciphertext = rsa.encrypt_bytes(message)
    assert rsa.decrypt_bytes(ciphertext) == message

    def do_keygen() -> None:
        ToyRSA(key_bits, rng)

    def do_encrypt() -> None:
        rsa.encrypt_bytes(message)

    def do_decrypt() -> None:
        rsa.decrypt_bytes(ciphertext)

    return {
        "scheme": "RSA",
        "tier": "%d-bit" % key_bits,
        "keygen_s": _time_per_call(do_keygen, number=1),
        "encrypt_s": _time_per_call(do_encrypt, number=20),
        "decrypt_s": _time_per_call(do_decrypt, number=20),
    }


def main() -> None:
    rng = random.Random(SEED)

    ntru_tiers = [
        (107, 512, 29, 10, 10),
        (167, 512, 45, 20, 20),
        (509, 2048, 216, 72, 72),
    ]
    rsa_tiers = [512, 1024, 2048]

    results = []
    print("Benchmarking NTRU tiers (seed=%d, message=%d bytes)..." % (SEED, MESSAGE_BYTES))
    for n, q, df, dg, dr in ntru_tiers:
        row = benchmark_ntru_tier(n, q, df, dg, dr, rng)
        results.append(row)
        print(
            "  NTRU %-12s keygen %9.4f s | encrypt %9.5f s | decrypt %9.5f s"
            % (row["tier"], row["keygen_s"], row["encrypt_s"], row["decrypt_s"])
        )

    print("Benchmarking RSA tiers...")
    for bits in rsa_tiers:
        row = benchmark_rsa_tier(bits, rng)
        results.append(row)
        print(
            "  RSA  %-12s keygen %9.4f s | encrypt %9.5f s | decrypt %9.5f s"
            % (row["tier"], row["keygen_s"], row["encrypt_s"], row["decrypt_s"])
        )

    # ------------------------------------------------------------------
    # Table output (console + CSV)
    # ------------------------------------------------------------------
    header = ["scheme", "tier", "keygen_ms", "encrypt_ms", "decrypt_ms"]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in results:
            writer.writerow(
                [
                    row["scheme"],
                    row["tier"],
                    "%.4f" % (row["keygen_s"] * 1000.0),
                    "%.4f" % (row["encrypt_s"] * 1000.0),
                    "%.4f" % (row["decrypt_s"] * 1000.0),
                ]
            )
    print("Results written to %s" % csv_path)

    print("%-6s %-12s %12s %12s %12s" % ("Scheme", "Tier", "Keygen (ms)", "Encrypt (ms)", "Decrypt (ms)"))
    print("-" * 58)
    for row in results:
        print(
            "%-6s %-12s %12.3f %12.3f %12.3f"
            % (
                row["scheme"],
                row["tier"],
                row["keygen_s"] * 1000.0,
                row["encrypt_s"] * 1000.0,
                row["decrypt_s"] * 1000.0,
            )
        )

    # ------------------------------------------------------------------
    # Chart output
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("keygen_s", "Key generation"), ("encrypt_s", "Encryption"), ("decrypt_s", "Decryption")]
    x_positions = np.arange(len(results))
    colors = ["#1f77b4" if row["scheme"] == "NTRU" else "#d62728" for row in results]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        values = [row[metric_key] * 1000.0 for row in results]
        bars = ax.bar(x_positions, values, color=colors)
        ax.set_yscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            ["%s %s" % (row["scheme"], row["tier"]) for row in results],
            fontsize=8,
        )
        ax.set_ylabel("Latency (ms, log scale)")
        ax.set_title(metric_title)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.grid(axis="y", which="both", alpha=0.3)

    fig.suptitle(
        "NTRU vs textbook RSA latency (seed=%d, %d-byte message, min of %d repeats)"
        % (SEED, MESSAGE_BYTES, REPEAT)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = os.path.join(out_dir, "benchmark_results.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print("Chart written to %s" % png_path)


if __name__ == "__main__":
    main()