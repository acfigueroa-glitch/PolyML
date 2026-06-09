"""Ed25519 request signing for the Polymarket US API.

Per docs.polymarket.us, every authenticated request carries three headers:

    X-PM-Access-Key   your Key ID (public identifier)
    X-PM-Timestamp    current time in milliseconds (within 30s of server time)
    X-PM-Signature    base64( Ed25519_sign( "{timestamp}{METHOD}{path}" ) )

The signed message concatenates the timestamp, the HTTP method, and the request
path. Some POST endpoints additionally fold the request body into the message;
we support that via ``include_body`` so the same signer covers both cases.
"""

from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ACCESS_KEY_HEADER = "X-PM-Access-Key"
TIMESTAMP_HEADER = "X-PM-Timestamp"
SIGNATURE_HEADER = "X-PM-Signature"


def _decode_secret(secret_key: str) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from the string Polymarket provides.

    Accepts:
      * base64 (std or urlsafe) of a 32-byte seed,
      * base64 of a 64-byte expanded key (seed||public — we take the seed),
      * hex of a 32-byte seed.
    """
    candidate = secret_key.strip()

    # Try base64 (standard then urlsafe), then hex.
    raw: bytes | None = None
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoded = decoder(_pad_b64(candidate))
            if len(decoded) in (32, 64):
                raw = decoded
                break
        except Exception:  # noqa: BLE001 - fall through to next decoder
            continue
    if raw is None:
        try:
            decoded = bytes.fromhex(candidate)
            if len(decoded) in (32, 64):
                raw = decoded
        except ValueError:
            raw = None

    if raw is None:
        raise ValueError(
            "Could not parse POLYMARKET_SECRET_KEY. Expected base64 or hex of a "
            "32-byte Ed25519 seed (or 64-byte expanded key)."
        )

    seed = raw[:32]
    return Ed25519PrivateKey.from_private_bytes(seed)


def _pad_b64(s: str) -> str:
    """Restore '=' padding that is often stripped from base64 strings."""
    return s + "=" * (-len(s) % 4)


class Ed25519Signer:
    """Holds the private key and produces signatures + auth headers."""

    def __init__(self, key_id: str, secret_key: str) -> None:
        if not key_id:
            raise ValueError("key_id is required")
        self.key_id = key_id
        self._private_key = _decode_secret(secret_key)

    @classmethod
    def from_credentials(cls, key_id: str | None, secret_key: str | None) -> "Ed25519Signer":
        if not key_id or not secret_key:
            raise ValueError(
                "Missing credentials. Set POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY "
                "(see .env.example)."
            )
        return cls(key_id, secret_key)

    def sign(self, message: str) -> str:
        """Return the base64-encoded Ed25519 signature of ``message``."""
        signature = self._private_key.sign(message.encode("utf-8"))
        return base64.b64encode(signature).decode("ascii")

    def headers(
        self,
        method: str,
        path: str,
        *,
        body: str | None = None,
        include_body: bool = False,
        timestamp_ms: int | None = None,
    ) -> dict[str, str]:
        """Build the three signed headers for a request.

        ``path`` should be the request path including any query string, e.g.
        ``/v1/portfolio/positions?limit=100``.
        """
        ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}"
        if include_body and body:
            message += body
        return {
            ACCESS_KEY_HEADER: self.key_id,
            TIMESTAMP_HEADER: ts,
            SIGNATURE_HEADER: self.sign(message),
        }


def build_auth_headers(
    signer: Ed25519Signer,
    method: str,
    path: str,
    *,
    body: str | None = None,
    include_body: bool = False,
) -> dict[str, str]:
    """Convenience wrapper used by the REST and WebSocket clients."""
    return signer.headers(method, path, body=body, include_body=include_body)
