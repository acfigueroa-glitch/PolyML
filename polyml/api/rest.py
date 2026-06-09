"""REST client for the Polymarket US API.

Wraps the documented endpoints we need to observe markets and your account:

  Public (no auth):
    GET  /markets, /markets/{slug}, /markets/{slug}/bbo
    GET  {gateway}/markets/{slug}/book
    GET  /events

  Authenticated (Ed25519 headers):
    GET  /accounts/who-am-i
    GET  /account/balances
    GET  /portfolio/positions
    GET  /portfolio/activities
    GET  /orders            (open orders)

The client is intentionally thin: it returns parsed JSON dicts. Higher layers
(collectors / mirror) decide what to persist. Only GET endpoints are exposed —
PolyML is observe-only and never places, modifies, or cancels orders.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

import httpx

from polyml.api.auth import Ed25519Signer

logger = logging.getLogger(__name__)


class PolymarketAPIError(RuntimeError):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, url: str, body: str) -> None:
        super().__init__(f"{status_code} for {url}: {body[:500]}")
        self.status_code = status_code
        self.url = url
        self.body = body


class RestClient:
    def __init__(
        self,
        rest_base_url: str,
        gateway_base_url: str,
        *,
        signer: Ed25519Signer | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.rest_base_url = rest_base_url.rstrip("/")
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.signer = signer
        self._client = client or httpx.Client(timeout=timeout)
        # The path prefix used when signing (everything after the host).
        self._rest_path_prefix = httpx.URL(self.rest_base_url).path.rstrip("/")

    # --- low-level ---------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RestClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        base: str | None = None,
        auth: bool = False,
    ) -> Any:
        base_url = (base or self.rest_base_url).rstrip("/")
        # Build the full path (with query) used both for the URL and for signing.
        query = ""
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                query = "?" + urlencode(clean, doseq=True)
        url = f"{base_url}{path}{query}"

        headers: dict[str, str] = {"Accept": "application/json"}
        if auth:
            if self.signer is None:
                raise PolymarketAPIError(401, url, "Authenticated call requires credentials")
            # Sign the host-relative path including the API version prefix.
            sign_path = httpx.URL(url).path
            if httpx.URL(url).query:
                sign_path += "?" + httpx.URL(url).query.decode()
            headers.update(self.signer.headers(method, sign_path))

        logger.debug("%s %s (auth=%s)", method, url, auth)
        resp = self._client.request(method, url, headers=headers)
        if resp.status_code >= 400:
            raise PolymarketAPIError(resp.status_code, url, resp.text)
        if not resp.content:
            return None
        return resp.json()

    def get(self, path: str, **kw: Any) -> Any:
        return self._request("GET", path, **kw)

    # --- public market data ------------------------------------------------------
    def list_markets(self, limit: int = 50, cursor: str | None = None) -> Any:
        return self.get("/markets", params={"limit": limit, "cursor": cursor})

    def get_market(self, slug: str) -> Any:
        return self.get(f"/markets/{slug}")

    def get_market_bbo(self, slug: str) -> Any:
        return self.get(f"/markets/{slug}/bbo")

    def get_market_book(self, slug: str) -> Any:
        """The order book lives on the gateway host per the docs."""
        return self.get(f"/markets/{slug}/book", base=self.gateway_base_url)

    def list_events(self, limit: int = 50, cursor: str | None = None) -> Any:
        return self.get("/events", params={"limit": limit, "cursor": cursor})

    # --- authenticated account / portfolio --------------------------------------
    def who_am_i(self) -> Any:
        return self.get("/accounts/who-am-i", auth=True)

    def get_balances(self) -> Any:
        return self.get("/account/balances", auth=True)

    def get_positions(self) -> Any:
        return self.get("/portfolio/positions", auth=True)

    def get_open_orders(self) -> Any:
        return self.get("/orders", auth=True)

    def get_activities(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        market_slug: str | None = None,
        types: list[str] | None = None,
        sort_order: str = "DESCENDING",
    ) -> Any:
        return self.get(
            "/portfolio/activities",
            auth=True,
            params={
                "limit": limit,
                "cursor": cursor,
                "marketSlug": market_slug,
                "types": types,
                "sortOrder": sort_order,
            },
        )

    def iter_activities(self, **kw: Any) -> Any:
        """Yield activity records across all pages (handles cursor pagination)."""
        cursor = kw.pop("cursor", None)
        while True:
            page = self.get_activities(cursor=cursor, **kw)
            if not page:
                return
            for activity in page.get("activities", []):
                yield activity
            if page.get("eof") or not page.get("nextCursor"):
                return
            cursor = page["nextCursor"]
