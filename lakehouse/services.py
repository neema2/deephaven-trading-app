"""
Lakehouse Service Managers — EmbeddedPG + Lakekeeper + ObjectStore
==================================================================
Auto-download, start, health-check, and stop Lakekeeper (Iceberg REST catalog)
and object storage (via ``objectstore.configure()``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from objectstore import ObjectStore

from store.pg_compat import get_server as get_pg_server

logger = logging.getLogger(__name__)


class EmbeddedPGManager:
    """
    Manages a PostgreSQL instance using pg_compat (architecture-aware).

    On x86, this typically uses pgserver (embedded).
    On ARM, this uses pixeltable-pgserver (system-shim).
    """

    def __init__(
        self,
        data_dir: str = "data/lakehouse/postgres",
        port: int = 5488,
        user: str = "postgres",
    ) -> None:
        self._data_dir = Path(data_dir).resolve()
        self._port = port
        self._user = user
        self._pg_server = None
        self._container_name = None
        self._pg_url_override = None

    @property
    def pg_url(self) -> str:
        """Connection URL for Lakekeeper."""
        if self._pg_url_override:
            return self._pg_url_override

        if not self._pg_server:
            return f"postgresql://{self._user}@127.0.0.1:{self._port}/postgres"

        uri = self._pg_server.get_uri()
        
        # Rust sqlx (Lakekeeper) fails parsing empty hostnames like "postgresql://user:@/dbname?host=..."
        # We inject localhost as a dummy hostname (the actual socket path is still honored via ?host=).
        import urllib.parse
        parsed = urllib.parse.urlparse(uri)
        if not parsed.hostname:
            netloc = parsed.netloc
            if netloc.endswith("@"):
                netloc += "localhost"
            parsed = parsed._replace(netloc=netloc)
            uri = urllib.parse.urlunparse(parsed)
            
        # Ensure database is postgres
        if not parsed.path or parsed.path == "/":
            uri = uri.rstrip("/") + "/postgres"
            
        return uri

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        """Start PG using the compatibility layer or Docker natively on ARM."""
        if platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
            import socket
            import uuid
            
            self._container_name = f"lk-pg-{uuid.uuid4().hex[:6]}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                self._port = s.getsockname()[1]
                
            logger.info("ARM64 detected. Starting native Docker PostgreSQL (%s) on port %d...", self._container_name, self._port)
            subprocess.run([
                "docker", "run", "-d", "--rm", "--name", self._container_name,
                "-p", f"{self._port}:5432",
                "-e", "POSTGRES_USER=postgres",
                "-e", "POSTGRES_PASSWORD=postgres",
                "-e", "POSTGRES_DB=postgres",
                "postgres:15"
            ], check=True, capture_output=True)

            # Wait for ready status
            for _ in range(30):
                res = subprocess.run(["docker", "exec", self._container_name, "pg_isready", "-U", "postgres"], capture_output=True)
                if res.returncode == 0:
                    break
                await asyncio.sleep(0.5)

            self._pg_url_override = f"postgresql://postgres:postgres@127.0.0.1:{self._port}/postgres"
            return

        try:
            from store.pg_compat import PGSERVER_FILE, ensure_pgcrypto_shim, ensure_uuid_ossp_shim
            ensure_uuid_ossp_shim(PGSERVER_FILE)
            ensure_pgcrypto_shim(PGSERVER_FILE)

            # Let the platform choose the port (it often uses static mapping based on data_dir)
            self._pg_server = get_pg_server(str(self._data_dir))
            
            # Update our port reference from the actual URI picked by the server.
            uri = self._pg_server.get_uri()
            import urllib.parse
            parsed = urllib.parse.urlparse(uri)
            if parsed.port:
                self._port = parsed.port
            logger.info("Embedded PG started/found on port %d", self._port)
        except Exception as e:
            logger.error("Failed to start PostgreSQL via pg_compat: %s", e)
            raise RuntimeError(f"Failed to start embedded PG: {e}") from e

    async def stop(self) -> None:
        """Stop PG via pg_compat or Docker container."""
        if self._container_name:
            subprocess.run(["docker", "stop", self._container_name], check=False, capture_output=True)
            self._container_name = None
            logger.info("Docker PG stopped")

        if self._pg_server:
            # pgserver uses cleanup() to stop the server
            self._pg_server.cleanup()
            self._pg_server = None
            logger.info("Embedded PG stopped")


    def _run_initdb(self) -> None:
        """Deprecated."""
        pass

    def _ensure_binaries(self) -> None:
        """Deprecated."""
        pass


def ensure_pgcrypto() -> bool:
    """Legacy helper — pgcrypto is included in zonkyio binaries by default.

    This function is kept for backward compatibility but is a no-op when using
    EmbeddedPGManager (which ships full contrib). Only useful if you're trying
    to add pgcrypto to pgserver's minimal build, which also lacks ICU and is
    therefore insufficient for Lakekeeper.
    """
    logger.debug("ensure_pgcrypto() called — zonkyio binaries include pgcrypto by default")
    return False


# ── Unified Lakehouse Lifecycle ────────────────────────────────────────────


class LakehouseStack:
    """Running lakehouse infrastructure.

    Properties:
        pg_url: PostgreSQL connection URL (for SyncEngine).
        catalog_url: Lakekeeper REST catalog URL.
        s3_endpoint: S3-compatible endpoint URL.
    """

    def __init__(
        self,
        pg: EmbeddedPGManager,
        lakekeeper: LakekeeperManager,
        s3_store: ObjectStore,
    ) -> None:
        self._pg = pg
        self._lakekeeper = lakekeeper
        self._s3_store = s3_store

    @property
    def pg_url(self) -> str:
        return self._pg.pg_url

    @property
    def catalog_url(self) -> str:
        return self._lakekeeper.catalog_url

    @property
    def s3_endpoint(self) -> str:
        return self._s3_store.endpoint


async def start_lakehouse(
    data_dir: str = "data/lakehouse",
    pg_port: int = 5488,
    lakekeeper_port: int = 8181,
    s3_api_port: int = 9002,
    s3_console_port: int = 9003,
    warehouse: str = "lakehouse",
    bucket: str = "lakehouse",
) -> LakehouseStack:
    """
    Start the full lakehouse stack: EmbeddedPG → ObjectStore → Lakekeeper.

    Downloads binaries on first run (~30s each for PG, Lakekeeper, object store).
    Subsequent starts are fast (< 5s total).

    Returns a LakehouseStack with all managers and connection info.
    """
    import objectstore

    pg = EmbeddedPGManager(data_dir=f"{data_dir}/postgres", port=pg_port)
    lk = LakekeeperManager(
        data_dir=f"{data_dir}/lakekeeper",
        port=lakekeeper_port,
    )

    # Start PG first (Lakekeeper depends on it)
    await pg.start()

    # Start object store (Lakekeeper warehouse depends on it)
    s3_store = await objectstore.configure(
        "minio",
        data_dir=f"{data_dir}/objectstore",
        api_port=s3_api_port,
        console_port=s3_console_port,
    )
    await s3_store.ensure_bucket(bucket)

    # Start Lakekeeper (depends on both PG and object store)
    await lk.start(pg_url=pg.pg_url)
    await lk.bootstrap()
    await lk.create_warehouse(
        name=warehouse,
        bucket=bucket,
        s3_endpoint=s3_store.endpoint,
    )

    logger.info(
        "Lakehouse stack running: PG=%d, Lakekeeper=%d, S3=%d",
        pg_port, lakekeeper_port, s3_api_port,
    )
    stack = LakehouseStack(pg=pg, lakekeeper=lk, s3_store=s3_store)
    return stack


async def stop_lakehouse(stack: LakehouseStack) -> None:
    """Stop all lakehouse services in reverse order."""
    await stack._lakekeeper.stop()
    # Object store cleanup handled by atexit
    await stack._pg.stop()
    logger.info("Lakehouse stack stopped")


# ── Lakekeeper ──────────────────────────────────────────────────────────────

LAKEKEEPER_VERSION = "0.11.2"
LAKEKEEPER_BASE_URL = "https://github.com/lakekeeper/lakekeeper/releases/download"

_LAKEKEEPER_TARGETS = {
    ("darwin", "arm64"): "lakekeeper-aarch64-apple-darwin.tar.gz",
    ("darwin", "aarch64"): "lakekeeper-aarch64-apple-darwin.tar.gz",
    ("darwin", "x86_64"): "lakekeeper-aarch64-apple-darwin.tar.gz",  # fallback
    ("linux", "x86_64"): "lakekeeper-x86_64-unknown-linux-gnu.tar.gz",
    ("linux", "aarch64"): "lakekeeper-aarch64-unknown-linux-gnu.tar.gz",
    ("linux", "arm64"): "lakekeeper-aarch64-unknown-linux-gnu.tar.gz",
}


def _lakekeeper_archive_name() -> str:
    """Detect the correct Lakekeeper archive for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    key = (system, machine)
    if key not in _LAKEKEEPER_TARGETS:
        raise RuntimeError(
            f"No Lakekeeper binary for {system}/{machine}. "
            "Use Docker instead: docker run -p 8181:8181 lakekeeper/lakekeeper"
        )
    return _LAKEKEEPER_TARGETS[key]


class LakekeeperManager:
    """Manages Lakekeeper binary lifecycle: download, migrate, serve, stop."""

    def __init__(
        self,
        data_dir: str = "data/lakehouse/lakekeeper",
        port: int = 8181,
        pg_url: str | None = None,
        encryption_key: str = "py-flow-dev-key-do-not-use-in-prod",
    ) -> None:
        self._data_dir = Path(data_dir).resolve()
        self._port = port
        self._pg_url = pg_url
        self._encryption_key = encryption_key
        self._process: subprocess.Popen | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def catalog_url(self) -> str:
        return f"http://localhost:{self._port}/catalog"

    async def start(self, pg_url: str | None = None) -> None:
        """Download if needed, run migrate, then start serving."""
        if await self.health():
            logger.info("Lakekeeper already running on port %d", self._port)
            return

        pg = pg_url or self._pg_url
        if not pg:
            raise ValueError(
                "Lakekeeper requires a PG connection URL. "
                "Pass pg_url or set LAKEKEEPER_PG_URL."
            )

        binary = self._ensure_binary()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["LAKEKEEPER__PG_DATABASE_URL_READ"] = pg
        env["LAKEKEEPER__PG_DATABASE_URL_WRITE"] = pg
        env["LAKEKEEPER__PG_ENCRYPTION_KEY"] = self._encryption_key
        env["LAKEKEEPER__LISTEN_PORT"] = str(self._port)
        env["LAKEKEEPER__PG_SSL_MODE"] = "disable"
        env["LAKEKEEPER__METRICS_PORT"] = "9100"

        # Run migrate first
        logger.info("Running Lakekeeper migrate...")
        migrate = subprocess.run(
            [str(binary), "migrate"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        if migrate.returncode != 0:
            stderr = migrate.stderr.decode()[:500]
            raise RuntimeError(f"Lakekeeper migrate failed: {stderr}")
        logger.info("Lakekeeper migrate complete")

        # Start serve
        logger.info("Starting Lakekeeper on port %d...", self._port)
        self._process = subprocess.Popen(
            [str(binary), "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for _attempt in range(30):
            await asyncio.sleep(1)
            if await self.health():
                logger.info("Lakekeeper started on port %d", self._port)
                return
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(f"Lakekeeper exited during startup: {stderr[:500]}")

        raise RuntimeError(f"Lakekeeper failed to start within 20s (port {self._port})")

    async def stop(self) -> None:
        """Gracefully stop Lakekeeper."""
        if self._process and self._process.poll() is None:
            logger.info("Stopping Lakekeeper (pid=%d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Lakekeeper did not stop gracefully, killing")
                self._process.kill()
                self._process.wait(timeout=5)
            logger.info("Lakekeeper stopped")
        self._process = None

    async def health(self) -> bool:
        """Check Lakekeeper health."""
        url = f"http://localhost:{self._port}/health"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def bootstrap(self) -> None:
        """Bootstrap Lakekeeper (first-time setup). Idempotent."""
        url = f"http://localhost:{self._port}/management/v1/bootstrap"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={"accept-terms-of-use": True})
                if resp.status_code in (200, 201, 409):
                    logger.info("Lakekeeper bootstrapped (status=%d)", resp.status_code)
                else:
                    logger.warning("Lakekeeper bootstrap returned %d: %s",
                                   resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Lakekeeper bootstrap failed: %s", e)

    async def create_warehouse(
        self,
        name: str = "lakehouse",
        bucket: str = "lakehouse",
        s3_endpoint: str = "http://localhost:9002",
        s3_access_key: str = "minioadmin",
        s3_secret_key: str = "minioadmin",
        s3_region: str = "us-east-1",
    ) -> None:
        """Create a warehouse in Lakekeeper pointing at S3 storage. Idempotent."""
        url = f"http://localhost:{self._port}/management/v1/warehouse"
        payload = {
            "warehouse-name": name,
            "project-id": None,
            "storage-profile": {
                "type": "s3",
                "bucket": bucket,
                "region": s3_region,
                "endpoint": s3_endpoint,
                "path-style-access": True,
                "flavor": "s3-compat",
                "sts-enabled": True,
            },
            "storage-credential": {
                "type": "s3",
                "credential-type": "access-key",
                "aws-access-key-id": s3_access_key,
                "aws-secret-access-key": s3_secret_key,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code in (200, 201):
                    logger.info("Lakekeeper warehouse '%s' created", name)
                elif resp.status_code == 409:
                    logger.info("Lakekeeper warehouse '%s' already exists", name)
                else:
                    logger.warning(
                        "Lakekeeper warehouse creation returned %d: %s",
                        resp.status_code, resp.text[:300],
                    )
        except Exception as e:
            logger.warning("Lakekeeper warehouse creation failed: %s", e)

    def _ensure_binary(self) -> Path:
        """Ensure Lakekeeper binary is available, downloading if needed."""
        bin_dir = self._data_dir / "bin"
        binary = bin_dir / "lakekeeper"
        if binary.exists() and os.access(binary, os.X_OK):
            return binary

        logger.info("Lakekeeper binary not found, downloading v%s...", LAKEKEEPER_VERSION)
        archive_name = _lakekeeper_archive_name()
        url = f"{LAKEKEEPER_BASE_URL}/v{LAKEKEEPER_VERSION}/{archive_name}"

        bin_dir.mkdir(parents=True, exist_ok=True)
        archive_path = bin_dir / archive_name

        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        logger.info("Downloaded %s", archive_name)

        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(path=bin_dir)

        archive_path.unlink()

        # Find the binary (may be in a subdirectory)
        if not binary.exists():
            for candidate in bin_dir.rglob("lakekeeper"):
                if candidate.is_file():
                    candidate.rename(binary)
                    break

        if binary.exists():
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        if not binary.exists():
            raise FileNotFoundError(f"Lakekeeper binary not found after extraction in {bin_dir}")

        logger.info("Lakekeeper v%s installed to %s", LAKEKEEPER_VERSION, binary)
        return binary


