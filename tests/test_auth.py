"""Tests for Ed25519 request signing."""

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from polyml.api.auth import (
    ACCESS_KEY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    Ed25519Signer,
)


def _make_signer() -> tuple[Ed25519Signer, Ed25519PrivateKey]:
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes_raw()
    secret_b64 = base64.b64encode(seed).decode()
    return Ed25519Signer("my-key-id", secret_b64), key


def test_headers_present_and_named_correctly():
    signer, _ = _make_signer()
    headers = signer.headers("GET", "/v1/portfolio/positions", timestamp_ms=1700000000000)
    assert headers[ACCESS_KEY_HEADER] == "my-key-id"
    assert headers[TIMESTAMP_HEADER] == "1700000000000"
    assert SIGNATURE_HEADER in headers


def test_signature_verifies_against_public_key():
    signer, key = _make_signer()
    ts = 1700000000000
    headers = signer.headers("GET", "/v1/account/balances", timestamp_ms=ts)
    message = f"{ts}GET/v1/account/balances".encode()
    signature = base64.b64decode(headers[SIGNATURE_HEADER])
    # Raises InvalidSignature if it doesn't verify.
    key.public_key().verify(signature, message)


def test_hex_secret_is_accepted():
    key = Ed25519PrivateKey.generate()
    seed_hex = key.private_bytes_raw().hex()
    signer = Ed25519Signer("kid", seed_hex)
    headers = signer.headers("GET", "/v1/markets", timestamp_ms=1)
    sig = base64.b64decode(headers[SIGNATURE_HEADER])
    key.public_key().verify(sig, b"1GET/v1/markets")


def test_include_body_changes_signature():
    signer, _ = _make_signer()
    a = signer.headers("POST", "/v1/orders", body='{"x":1}', include_body=True, timestamp_ms=1)
    b = signer.headers("POST", "/v1/orders", body=None, include_body=False, timestamp_ms=1)
    assert a[SIGNATURE_HEADER] != b[SIGNATURE_HEADER]
