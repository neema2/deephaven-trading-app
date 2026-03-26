"""
streaming.admin — Platform API for Real-Time Streaming
=======================================================
Start/stop the ticking table server, register aliases.

Platform usage::

    from streaming.admin import StreamingServer

    server = StreamingServer(port=10000)
    server.start()
    server.register_alias("demo")

Hides Deephaven as an implementation detail.

On ARM platforms (aarch64/arm64), Deephaven's bundled JVM is x86-only,
so we start the server via Docker instead.  This is architecture-gated —
Docker is NEVER used on x86 as a fallback.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Any

from streaming._registry import register_alias as _register_alias

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture detection — the ONLY gate for Docker vs in-process
# ---------------------------------------------------------------------------

_ARM_MACHINES = {"aarch64", "arm64", "ARM64"}


def is_arm() -> bool:
    """Return True if running on an ARM64 CPU."""
    return platform.machine() in _ARM_MACHINES


def _needs_docker() -> bool:
    """Return True only on Linux ARM64 where Deephaven's x86 JVM won't run.

    macOS ARM uses Rosetta 2 to run the bundled x86 JVM — no Docker needed.
    Windows ARM is unsupported (no Docker on GitHub's windows-11-arm runner).

    Set ``FORCE_DOCKER_STREAMING=1`` to force Docker mode on any platform
    (useful for local testing of the remote code path).
    """
    if os.environ.get("FORCE_DOCKER_STREAMING") == "1":
        return True
    return platform.system() == "Linux" and platform.machine() in _ARM_MACHINES


# ---------------------------------------------------------------------------
# StreamingServer
# ---------------------------------------------------------------------------


class StreamingServer:
    """Real-time ticking table server.

    On x86: wraps the in-process Deephaven JVM (``deephaven_server.Server``).
    On ARM: starts ``ghcr.io/deephaven/server:latest`` via Docker.
    """

    def __init__(
        self,
        port: int = 10000,
        max_heap: str = "1g",
        *,
        jvm_args: list[str] | None = None,
        default_jvm_args: list[str] | None = None,
    ) -> None:
        self._port = port
        self._max_heap = max_heap
        self._jvm_args = jvm_args or [
            f"-Xmx{max_heap}",
            "-Dprocess.info.system-info.enabled=false",
            "-DAuthHandlers=io.deephaven.auth.AnonymousAuthenticationHandler",
        ]
        self._default_jvm_args = default_jvm_args or [
            "-XX:+UseG1GC",
            "-XX:MaxGCPauseMillis=100",
            "-XX:+UseStringDeduplication",
        ]
        self._server: Any = None  # in-process Server or None
        self._container_id: str | None = None  # Docker container ID
        self._is_remote = _needs_docker()

    def start(self) -> StreamingServer:
        """Start the streaming server."""
        if self._is_remote:
            self._start_docker()
        else:
            self._start_in_process()
        return self

    def _start_in_process(self) -> None:
        """Start Deephaven JVM in-process (x86 only)."""
        from deephaven_server import Server

        self._server = Server(
            port=self._port,
            jvm_args=self._jvm_args,
            default_jvm_args=self._default_jvm_args,
        )
        self._server.start()
        logger.info("StreamingServer started in-process on port %d", self._port)

    def _start_docker(self) -> None:
        """Start Deephaven via Docker container (ARM only)."""
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError(
                "Docker is required for Deephaven on ARM but 'docker' "
                "was not found on PATH."
            )

        container_name = f"dh-streaming-{self._port}"

        # Stop any leftover container from a previous run
        subprocess.run(
            [docker, "rm", "-f", container_name],
            capture_output=True,
        )

        cmd = [
            docker, "run", "-d",
            "--name", container_name,
            "-p", f"{self._port}:{self._port}",
            "-e", f"START_OPTS=-Xmx{self._max_heap} "
                  f"-DAuthHandlers=io.deephaven.auth.AnonymousAuthenticationHandler "
                  f"-Ddeephaven.console.type=python",
            "ghcr.io/deephaven/server:latest",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start Deephaven Docker container: {result.stderr}"
            )
        self._container_id = result.stdout.strip()

        # Health-check: wait for the gRPC port to accept connections
        self._wait_for_ready(timeout=60)
        logger.info(
            "StreamingServer started via Docker (container=%s) on port %d",
            container_name, self._port,
        )

    def _wait_for_ready(self, timeout: int = 60) -> None:
        """Block until Deephaven is accepting pydeephaven sessions."""
        from pydeephaven import Session

        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = Session(host="localhost", port=self._port)
                s.close()
                return
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        raise RuntimeError(
            f"Deephaven Docker container not ready after {timeout}s: {last_err}"
        )

    def stop(self) -> None:
        """Stop the streaming server."""
        if self._container_id is not None:
            docker = shutil.which("docker")
            if docker:
                container_name = f"dh-streaming-{self._port}"
                subprocess.run(
                    [docker, "rm", "-f", container_name],
                    capture_output=True,
                )
            self._container_id = None
            logger.info("StreamingServer Docker container stopped")
        elif self._server is not None:
            self._server = None
            logger.info("StreamingServer stopped")

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def remote(self) -> bool:
        """True if this server runs in Docker (ARM mode)."""
        return self._is_remote

    def register_alias(self, name: str) -> None:
        """Register this server under an alias name."""
        _register_alias(name, port=self._port)

    def __enter__(self) -> StreamingServer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


__all__ = ["StreamingServer", "is_arm"]
