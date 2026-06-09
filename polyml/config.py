"""Configuration loading for PolyML.

Precedence (highest wins):
    1. Explicit overrides passed to ``load_config``.
    2. Environment variables (and a ``.env`` file if present).
    3. ``config/local.yaml`` if it exists.
    4. ``config/default.yaml``.

Credentials live ONLY in the environment (never in YAML), so secrets are not
accidentally committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"


# --- Environment variable mapping -------------------------------------------------
# Maps an env var to a dotted path inside the config dict.
_ENV_OVERRIDES: dict[str, str] = {
    "POLYML_REST_BASE_URL": "api.rest_base_url",
    "POLYML_GATEWAY_BASE_URL": "api.gateway_base_url",
    "POLYML_WS_PRIVATE_URL": "api.ws_private_url",
    "POLYML_WS_MARKETS_URL": "api.ws_markets_url",
    "POLYML_DB_PATH": "storage.db_path",
    "POLYML_LOG_LEVEL": "logging.level",
}


@dataclass(frozen=True)
class Credentials:
    """Polymarket US API credentials, sourced from the environment only."""

    key_id: str | None = None
    secret_key: str | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.key_id) and bool(self.secret_key)


@dataclass
class Config:
    """Resolved configuration. ``raw`` holds the merged YAML dict for free-form
    access; the typed helpers cover the common paths."""

    raw: dict[str, Any] = field(default_factory=dict)
    credentials: Credentials = field(default_factory=Credentials)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # Convenience accessors -------------------------------------------------------
    @property
    def db_path(self) -> Path:
        p = Path(self.get("storage.db_path", "data/polyml.db"))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def rest_base_url(self) -> str:
        return self.get("api.rest_base_url")

    @property
    def gateway_base_url(self) -> str:
        return self.get("api.gateway_base_url")

    @property
    def ws_private_url(self) -> str:
        return self.get("api.ws_private_url")

    @property
    def ws_markets_url(self) -> str:
        return self.get("api.ws_markets_url")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency). Does not overwrite vars
    already present in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(overrides: dict[str, Any] | None = None) -> Config:
    """Load and merge configuration from all sources."""
    _load_dotenv(PROJECT_ROOT / ".env")

    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    if LOCAL_CONFIG_PATH.exists():
        raw = _deep_merge(raw, yaml.safe_load(LOCAL_CONFIG_PATH.read_text()) or {})

    # Apply env var overrides for known config paths.
    for env_var, dotted in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            _set_dotted(raw, dotted, os.environ[env_var])

    if overrides:
        raw = _deep_merge(raw, overrides)

    credentials = Credentials(
        key_id=os.environ.get("POLYMARKET_KEY_ID") or None,
        secret_key=os.environ.get("POLYMARKET_SECRET_KEY") or None,
    )
    return Config(raw=raw, credentials=credentials)
