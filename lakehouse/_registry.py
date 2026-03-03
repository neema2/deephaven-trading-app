"""
Lakehouse alias registry — maps alias names to lakehouse connection info.
"""

from __future__ import annotations

import threading

_aliases: dict[str, dict] = {}
_lock = threading.Lock()


def register_alias(name: str, catalog_url: str, s3_endpoint: str,
                   s3_access_key: str = "minioadmin", s3_secret_key: str = "minioadmin",
                   s3_region: str = "us-east-1", warehouse: str = "lakehouse",
                   namespace: str = "default") -> None:
    """Register a lakehouse server alias."""
    with _lock:
        _aliases[name] = {
            "catalog_url": catalog_url,
            "s3_endpoint": s3_endpoint,
            "s3_access_key": s3_access_key,
            "s3_secret_key": s3_secret_key,
            "s3_region": s3_region,
            "warehouse": warehouse,
            "namespace": namespace,
        }


def resolve_alias(name: str) -> dict | None:
    """Resolve a lakehouse alias to connection info."""
    with _lock:
        return _aliases.get(name)
