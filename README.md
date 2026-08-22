# Post-Quantum Lattice Cryptography: NTRU over Polynomial Quotient Rings

Final Year Project package for a BSc in Pure Mathematics and Computer Science.
Implements the NTRU cryptosystem from first principles over
$R = \mathbb{Z}[X]/(X^N - 1)$, wraps it in a FastAPI microservice, validates it
with a seeded correctness and security test suite, and benchmarks it against a
from-scratch textbook RSA baseline.

The full mathematical specification — quotient ring algebra, key generation,
ideal mapping, encryption/decryption, and the decryption failure analysis — is
in [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Repository layout

```
ntru_project/
├── docs/
│   └── PROPOSAL.md            # Academic formal proposal (Sections 1-5)
├── ntru/
│   ├── __init__.py
│   ├── engine.py              # NTRUEngine: ring algebra, EEA mod p, Hensel lifting mod q
│   ├── encoding.py            # TernaryCodec: bytes <-> ternary message polynomials
│   └── toy_rsa.py             # Textbook RSA baseline (benchmarking only; NOT secure)
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI microservice (/keygen, /encrypt, /decrypt, /health)
├── tests/
│   └── test_ntru.py           # unittest suite: correctness, failure rate, validation
├── benchmarks/
│   └── benchmark.py           # timeit + matplotlib NTRU vs RSA suite
├── requirements.txt
└── README.md
```

## Installation

```
pip install -r requirements.txt
```

Requires Python 3.9+ (uses `pow(x, -1, m)` modular inversion).

## Running the test suite

```
python -m unittest discover -s tests -v
```

The suite includes:

- `test_round_trip_correctness` — 200 seeded iterations (`random.seed(42)`-style
  deterministic `random.Random(42)`) of fresh key generation, random plaintext,
  encrypt, decrypt, asserting exact byte equality.
- `test_round_trip_production_parameters` — seeded round trips at the full
  academic parameter set (N=509, p=3, q=2048).
- `test_empirical_decryption_failure_rate_boundary_parameters` — 4000 seeded
  encrypt/decrypt cycles at deliberately stressed parameters (N=53, q=64),
  reporting the empirical failure rate against the theoretical bound
  Pr[failure] <= 2N exp(-q^2 / (8 sigma_C^2)) derived in the proposal.
- `test_production_parameter_batch_no_failures` — 300 cycles at the academic
  parameter set; zero failures expected (bound < 10^-300).
- `test_parameter_validation` — non-prime N, gcd(p,q) != 1, non-power-of-two q,
  and out-of-range Hamming weights are all rejected.
- Codec unit tests, including corruption detection and empty-message handling.

## Running the API service

```
uvicorn api.main:app --reload
```

Interactive OpenAPI documentation is then available at
`http://127.0.0.1:8000/docs`.

Endpoints:

- `POST /keygen` — returns `public_key` and, in a clearly separate
  `private_key` field, the private triple. Returning private key material over
  an API is for LOCAL DEMONSTRATION ONLY and is not a production-safe pattern
  (see the security notice in `api/main.py`).
- `POST /encrypt` — body: `{"plaintext": "...", "public_key": {...}}`;
  returns ciphertext polynomial coefficient blocks.
- `POST /decrypt` — body: `{"ciphertext_blocks": [[...]], "private_key": {...}}`;
  returns the decrypted plaintext string.
- `GET /health` — liveness probe with the active parameter set.

Example round trip with `curl`:

```
curl -s -X POST http://127.0.0.1:8000/keygen > keys.json
# build encrypt request from keys.json public_key, then:
curl -s -X POST http://127.0.0.1:8000/encrypt -H "Content-Type: application/json" -d @encrypt_req.json
# build decrypt request from the ciphertext and keys.json private_key, then:
curl -s -X POST http://127.0.0.1:8000/decrypt -H "Content-Type: application/json" -d @decrypt_req.json
```

## Running the benchmark suite

```
python -m benchmarks.benchmark
```

Produces:

- a console table of keygen/encrypt/decrypt latencies,
- `benchmarks/benchmark_results.csv`,
- `benchmarks/benchmark_results.png` (log-scale grouped bar chart).

Tiers: NTRU (N=107, q=512), (N=167, q=512), (N=509, q=2048) versus textbook
RSA at 512, 1024, and 2048 bits. Fixed seed 20260822; fixed 64-byte workload;
minimum-of-5-repeats `timeit` aggregation.

## Default parameter set

| Parameter | Value |
|---|---|
| N | 509 (prime) |
| p | 3 |
| q | 2048 = 2^11 |
| d_f | 216 (+1) / 215 (-1) |
| d_g, d_r | 72 / 72 |
| Bytes per message polynomial | 84 |

## Security status (read this)

- This is an academic reference implementation. It is not audited, not
  constant-time, and must not protect real secrets.
- Raw NTRU encryption as implemented is IND-CPA secure only under the
  CONJECTURED hardness of the underlying lattice problems. No proof of
  quantum hardness exists for any lattice problem; quantum resistance of NTRU
  is a conjecture supported by cryptanalytic evidence, not a theorem.
- IND-CCA security requires a transform (NAEP or Fujisaki-Okamoto), which is
  out of scope and listed as future work in the proposal.
- The toy RSA module is textbook RSA without padding: deterministic,
  malleable, and broken by Shor's algorithm on a large quantum computer. It
  exists solely as the benchmarking control group.