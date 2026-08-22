"""FastAPI microservice exposing the NTRU cryptosystem over HTTP.

Endpoints
=========
* POST /keygen  -- generate a key pair.
* POST /encrypt -- encrypt a UTF-8 plaintext string under a public key.
* POST /decrypt -- decrypt ciphertext blocks under a private key.
* GET  /health  -- liveness probe.

SECURITY NOTICE (READ BEFORE DEPLOYING)
=======================================
The /keygen endpoint returns the PRIVATE KEY in a dedicated response field.
This is intentional for local demonstration and grading workflows only. It
is emphatically NOT a production-safe pattern: private key material must
never transit a network interface in plaintext, never be logged, and never
be returned to any client that did not generate it inside a hardware or
attested enclave. A production deployment would keep the private key
server-side behind an authenticated decryption oracle with rate limiting,
or deliver it exclusively through an out-of-band secure channel.

This service performs no transport security itself; it must be terminated
behind TLS (e.g. via a reverse proxy) in any non-loopback deployment.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ntru.engine import (
    NTRUEngine,
    PrivateKey,
    PublicKey,
)
from ntru.encoding import TernaryCodec

app = FastAPI(
    title="NTRU Post-Quantum Cryptography Service",
    description=(
        "Academic microservice wrapping the NTRU lattice-based cryptosystem "
        "over R = Z[X]/(X^N - 1). See docs/PROPOSAL.md for the mathematical "
        "specification."
    ),
    version="1.0.0",
)

# Single engine instance bound to the default academic parameter set
# (N=509, p=3, q=2048, d_f=216, d_g=d_r=72).
_ENGINE = NTRUEngine()
_CODEC = TernaryCodec(_ENGINE.n)
# Operational randomness: OS-entropy seeded, NOT reproducible by design.
# Reproducibility is confined to the offline test suite, which seeds its own
# isolated random.Random instances.
_OPERATIONAL_RNG = random.SystemRandom()


# ----------------------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------------------


class KeygenResponse(BaseModel):
    """Response schema for /keygen."""

    public_key: dict = Field(..., description="Public key h = p*f^{-1}*g mod q.")
    private_key: dict = Field(
        ...,
        description=(
            "Private key (f, f_p^{-1}, f_q^{-1}). WARNING: returning private "
            "key material from an API is for LOCAL DEMONSTRATION ONLY and is "
            "not a production-safe pattern. See module docstring."
        ),
    )


class EncryptRequest(BaseModel):
    """Request schema for /encrypt."""

    plaintext: str = Field(..., description="UTF-8 plaintext string to encrypt.")
    public_key: dict = Field(..., description="Public key as returned by /keygen.")


class EncryptResponse(BaseModel):
    """Response schema for /encrypt."""

    block_count: int = Field(..., description="Number of ciphertext polynomials.")
    bytes_per_block: int = Field(
        ..., description="Plaintext bytes carried per message polynomial."
    )
    ciphertext_blocks: List[List[int]] = Field(
        ..., description="Ciphertext polynomial coefficients, each block length N."
    )


class DecryptRequest(BaseModel):
    """Request schema for /decrypt."""

    ciphertext_blocks: List[List[int]] = Field(
        ..., description="Ciphertext polynomial coefficients from /encrypt."
    )
    private_key: dict = Field(..., description="Private key as returned by /keygen.")


class DecryptResponse(BaseModel):
    """Response schema for /decrypt."""

    plaintext: str = Field(..., description="Decrypted UTF-8 plaintext string.")


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    parameters: dict


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe reporting the active parameter set."""
    return HealthResponse(
        status="ok",
        parameters={
            "n": _ENGINE.n,
            "p": _ENGINE.p,
            "q": _ENGINE.q,
            "df": _ENGINE.df,
            "dg": _ENGINE.dg,
            "dr": _ENGINE.dr,
            "bytes_per_block": _CODEC.bytes_per_block,
        },
    )


@app.post("/keygen", response_model=KeygenResponse)
def keygen() -> KeygenResponse:
    """Generate a fresh NTRU key pair.

    Returns the public key h = p * f^{-1} * g (mod q) and, in a clearly
    separate ``private_key`` field, the triple (f, f_p^{-1}, f_q^{-1}).

    SECURITY FLAG: the presence of ``private_key`` in this response exists
    solely so that a local demonstrator can copy it into /decrypt without a
    database. Returning private keys over an API is not a production-safe
    pattern; see the module-level security notice.
    """
    public_key, private_key = _ENGINE.generate_keypair(_OPERATIONAL_RNG)
    return KeygenResponse(
        public_key=public_key.to_dict(),
        private_key=private_key.to_dict(),
    )


@app.post("/encrypt", response_model=EncryptResponse)
def encrypt(request: EncryptRequest) -> EncryptResponse:
    """Encrypt a plaintext string under a public key.

    Pipeline: UTF-8 encode the string, apply the ternary codec of
    :mod:`ntru.encoding` (4-byte length prefix, base-3 digit expansion,
    balanced ternary coefficient mapping, zero padding of the final block),
    then encrypt each message polynomial as e = r*h + m (mod q) with fresh
    per-block blinding polynomial r.
    """
    try:
        public_key = PublicKey.from_dict(request.public_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if (public_key.n, public_key.p, public_key.q) != (_ENGINE.n, _ENGINE.p, _ENGINE.q):
        raise HTTPException(
            status_code=400,
            detail=(
                "Public key parameters (%d, %d, %d) do not match the service "
                "parameter set (%d, %d, %d)."
                % (
                    public_key.n,
                    public_key.p,
                    public_key.q,
                    _ENGINE.n,
                    _ENGINE.p,
                    _ENGINE.q,
                )
            ),
        )

    plaintext_bytes = request.plaintext.encode("utf-8")
    try:
        message_blocks = _CODEC.encode(plaintext_bytes)
        ciphertext_blocks = [
            _ENGINE.encrypt(m, public_key, _OPERATIONAL_RNG)
            for m in message_blocks
        ]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EncryptResponse(
        block_count=len(ciphertext_blocks),
        bytes_per_block=_CODEC.bytes_per_block,
        ciphertext_blocks=[
            [int(c) for c in block] for block in ciphertext_blocks
        ],
    )


@app.post("/decrypt", response_model=DecryptResponse)
def decrypt(request: DecryptRequest) -> DecryptResponse:
    """Decrypt ciphertext blocks under a private key.

    Pipeline: rebuild the private key object from the submitted JSON,
    decrypt each ciphertext polynomial via a = f*e (mod q), centered
    lifting, reduction mod p, multiplication by f_p^{-1}; then reverse the
    ternary codec to recover the original UTF-8 string.
    """
    try:
        private_key = PrivateKey.from_dict(request.private_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if (private_key.n, private_key.p, private_key.q) != (
        _ENGINE.n,
        _ENGINE.p,
        _ENGINE.q,
    ):
        raise HTTPException(
            status_code=400,
            detail="Private key parameters do not match the service parameter set.",
        )

    if len(request.ciphertext_blocks) == 0:
        raise HTTPException(status_code=400, detail="No ciphertext blocks supplied.")

    recovered_blocks: List[np.ndarray] = []
    for index, block in enumerate(request.ciphertext_blocks):
        if not isinstance(block, list) or len(block) != _ENGINE.n:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Ciphertext block %d must be a list of exactly N=%d integer "
                    "coefficients." % (index, _ENGINE.n)
                ),
            )
        try:
            arr = np.array([int(c) for c in block], dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Ciphertext block %d contains non-integer data." % index,
            ) from exc
        if np.any(arr < 0) or np.any(arr >= _ENGINE.q):
            raise HTTPException(
                status_code=400,
                detail="Ciphertext block %d has coefficients outside [0, q)." % index,
            )
        recovered_blocks.append(_ENGINE.decrypt(arr, private_key))

    try:
        plaintext_bytes = _CODEC.decode(recovered_blocks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DecryptResponse(plaintext=plaintext_bytes.decode("utf-8"))