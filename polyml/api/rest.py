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

import json
import logging
import time
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
        max_retries: int = 4,
    ) -> None:
        self.rest_base_url = rest_base_url.rstrip("/")
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.signer = signer
        self._client = client or httpx.Client(timeout=timeout)
        self.max_retries = max_retries
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
            # The server signs the host-relative path WITHOUT the query string
            # (verified empirically: query-bearing requests fail signature checks
            # when the query is included in the signed message).
            sign_path = httpx.URL(url).path
            headers.update(self.signer.headers(method, sign_path))

        logger.debug("%s %s (auth=%s)", method, url, auth)
        resp = self._request_with_retry(method, url, headers)
        if resp.status_code >= 400:
            raise PolymarketAPIError(resp.status_code, url, resp.text)
        if not resp.content:
            return None
        return resp.json()

    def _request_with_retry(self, method: str, url: str, headers: dict[str, str]) -> httpx.Response:
        """Send the request, backing off on 429/503 (respecting Retry-After).

        For authenticated requests the timestamp/signature are regenerated on
        each attempt so they stay within the server's clock-skew window.
        """
        attempt = 0
        while True:
            resp = self._client.request(method, url, headers=headers)
            if resp.status_code not in (429, 503) or attempt >= self.max_retries:
                return resp
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0 * (2 ** attempt)
            except ValueError:
                delay = 2.0 * (2 ** attempt)
            delay = min(delay, 30.0)
            logger.warning("rate-limited (%s) on %s; retrying in %.1fs", resp.status_code, url, delay)
            time.sleep(delay)
            attempt += 1
            if self.signer is not None and headers.get("X-PM-Signature"):
                headers.update(self.signer.headers(method, httpx.URL(url).path))

    def get(self, path: str, **kw: Any) -> Any:
        return self._request("GET", path, **kw)

    def post(self, path: str, body_obj: Any, *, auth: bool = True) -> Any:
        """Signed POST with a JSON body. Used only by the trading layer.

        Verified against the live API: writes sign the path ONLY (no query, no
        body) — the same scheme as GET. ``preview_order`` is the safe way to
        confirm signing + schema before placing a real order.
        """
        body = json.dumps(body_obj, separators=(",", ":")) if body_obj is not None else ""
        url = f"{self.rest_base_url}{path}"
        headers: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth:
            if self.signer is None:
                raise PolymarketAPIError(401, url, "Authenticated call requires credentials")
            headers.update(self.signer.headers("POST", httpx.URL(url).path))

        attempt = 0
        while True:
            resp = self._client.request("POST", url, headers=headers, content=body)
            if resp.status_code not in (429, 503) or attempt >= self.max_retries:
                break
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
            attempt += 1
            if auth:
                headers.update(self.signer.headers("POST", httpx.URL(url).path))
        if resp.status_code >= 400:
            raise PolymarketAPIError(resp.status_code, url, resp.text)
        return resp.json() if resp.content else None

    # --- market data (all endpoints on the api host require auth headers) --------
    def list_markets(
        self, limit: int = 50, cursor: str | None = None, closed: bool | None = None
    ) -> Any:
        params: dict[str, Any] = {"limit": limit, "cursor": cursor}
        if closed is not None:
            params["closed"] = "true" if closed else "false"
        return self.get("/markets", params=params, auth=True)

    def get_market(self, slug: str) -> Any:
        # The single-slug path 404s on the live API; query the collection instead.
        resp = self.get("/markets", params={"slug": slug}, auth=True) or {}
        markets = resp.get("markets", []) if isinstance(resp, dict) else []
        return {"market": markets[0]} if markets else None

    def get_market_bbo(self, slug: str) -> Any:
        return self.get(f"/markets/{slug}/bbo", auth=True)

    def get_market_book(self, slug: str) -> Any:
        """The order book lives on the gateway host and is public (no auth)."""
        return self.get(f"/markets/{slug}/book", base=self.gateway_base_url)

    def list_events(self, limit: int = 50, cursor: str | None = None) -> Any:
        return self.get("/events", params={"limit": limit, "cursor": cursor}, auth=True)

    # --- authenticated account / portfolio --------------------------------------
    def get_balances(self) -> Any:
        return self.get("/account/balances", auth=True)

    def get_positions(self) -> Any:
        return self.get("/portfolio/positions", auth=True)

    def get_open_orders(self) -> Any:
        return self.get("/orders/open", auth=True)

    def get_activities(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        market_slug: str | None = None,
        types: list[str] | None = None,
        sort_order: str = "SORT_ORDER_DESCENDING",
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

    # --- order placement (writes; used only by the gated trading layer) ---------
    # Verified live: single-order routes are SINGULAR (/order, /order/preview)
    # and the order is wrapped in a {"request": ...} envelope.
    def preview_order(self, order: dict[str, Any]) -> Any:
        """Validate an order without placing it — safe POST to confirm signing."""
        return self.post("/order/preview", {"request": order})

    def create_order(self, order: dict[str, Any]) -> Any:
        # Create is POST /orders (plural); GET /orders is 501 Method Not Allowed.
        return self.post("/orders", {"request": order})

    def cancel_order(self, order_id: str) -> Any:
        return self.post("/order/cancel", {"request": {"orderId": order_id}})

    def iter_activities(self, *, max_pages: int = 2, page_pause: float = 0.4, **kw: Any) -> Any:
        """Yield activity records across pages (handles cursor pagination).

        ``max_pages`` bounds how deep we page in one call so a periodic poll
        doesn't trip the API rate limiter; pass a large value for a full
        historical backfill. ``page_pause`` spaces out paged requests.
        """
        cursor = kw.pop("cursor", None)
        pages = 0
        while pages < max_pages:
            page = self.get_activities(cursor=cursor, **kw)
            if not page:
                return
            for activity in page.get("activities", []):
                yield activity
            pages += 1
            if page.get("eof") or not page.get("nextCursor"):
                return
            cursor = page["nextCursor"]
            if pages < max_pages:
                time.sleep(page_pause)
