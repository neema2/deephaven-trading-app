"""
streaming.port_check — Port occupancy checker with actionable diagnostics.

Checks whether a set of TCP ports are free before starting servers.
When a port is busy it identifies the owning process (name + PID) and
prints a ready-to-paste ``kill`` command so the user can free it.

Typical use — one import, one call::

    from streaming.port_check import preflight_check

    # In a demo (warns and continues by default):
    preflight_check({10000: "Deephaven server", 8000: "market data server"})

    # In a test (raises PortInUseError if any port is busy):
    preflight_check({10000: "Deephaven server", 8000: "market data server"}, strict=True)

Lower-level primitives are also available for custom logic::

    from streaming.port_check import probe_ports, check_ports, assert_ports_free

The module uses only the Python standard library (``socket``, ``subprocess``,
``os``) so it works on any platform without extra dependencies.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class PortInfo:
    """Details about a single port occupancy query."""
    port: int
    is_free: bool
    pid: int | None = None
    process_name: str | None = None
    cmdline: str | None = None
    kill_suggestion: str | None = None


class PortInUseError(RuntimeError):
    """Raised by assert_ports_free() when one or more ports are occupied."""

    def __init__(self, busy: list[PortInfo]) -> None:
        self.busy = busy
        lines = ["\n\nPort(s) already in use — please free them first:\n"]
        for info in busy:
            lines.append(_format_port_info(info))
        super().__init__("".join(lines))


# ── Internal helpers ─────────────────────────────────────────────────────────

def _is_port_free(port: int) -> bool:
    """Return True if the port is not bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_port_owner(port: int) -> tuple[int | None, str | None, str | None]:
    """Return (pid, process_name, cmdline) for the process holding *port*.

    Tries ``ss`` first (Linux, fast), then ``lsof`` (macOS / older Linux),
    then ``docker ps`` (catches containers whose port-forward hides the PID).
    Returns (None, None, None) if the owner cannot be determined.
    """
    # --- ss (iproute2) ---
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if f":{port}" in line and "pid=" in line:
                pid_part = line.split("pid=", 1)[1].split(",")[0]
                pid = int(pid_part)
                name_part = line.split('(("', 1)[1].split('"', 1)[0] if '(("' in line else "unknown"
                return pid, name_part, _cmdline(pid)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        pass

    # --- lsof ---
    try:
        out = subprocess.check_output(
            ["lsof", "-iTCP", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                return pid, name, _cmdline(pid)
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        pass

    # --- Docker (port may be owned by a container's port-forward, not a local PID) ---
    docker_name, docker_cmd = _find_docker_container(port)
    if docker_name:
        # Use sentinel pid=0 to signal "known owner but no local PID"
        return 0, f"docker:{docker_name}", docker_cmd

    return None, None, None


def _find_docker_container(port: int) -> tuple[str | None, str | None]:
    """Return (container_name, suggested_stop_cmd) if a Docker container uses *port*."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"publish={port}",
             "--format", "{{.Names}}\t{{.Image}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if parts:
                name = parts[0].strip()
                image = parts[1].strip() if len(parts) > 1 else ""
                if name:
                    return name, f"docker container running: {image}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None, None


def _cmdline(pid: int) -> str | None:
    """Read /proc/<pid>/cmdline on Linux (best-effort)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return None


def _format_port_info(info: PortInfo) -> str:
    lines = [f"\n  Port {info.port}: IN USE"]
    if info.pid is not None:  # pid=0 is the Docker sentinel — still show
        lines.append(f"    Process : {info.process_name or 'unknown'} (PID {info.pid if info.pid else 'n/a'})")
    if info.cmdline:
        cmd = info.cmdline if len(info.cmdline) <= 120 else info.cmdline[:117] + "..."
        lines.append(f"    Command : {cmd}")
    if info.kill_suggestion:
        lines.append(f"    To free  : {info.kill_suggestion}")
    return "\n".join(lines) + "\n"


def _make_kill_suggestion(pid: int | None, process_name: str | None, cmdline: str | None) -> str | None:
    """Return a ready-to-paste shell command to free the port."""
    if pid is None:
        return None

    # Docker container — process_name is "docker:<container_name>"
    if process_name and process_name.startswith("docker:"):
        container = process_name[len("docker:"):]
        return f"docker stop {container}"

    # Any Python script — suggest pkill by script name
    if cmdline:
        script = next((w for w in cmdline.split() if w.endswith(".py")), None)
        if script:
            basename = os.path.basename(script)
            return f"pkill -f {basename}   # or: kill {pid}"
        # uvicorn / module invocations
        if "-m" in cmdline.split():
            idx = cmdline.split().index("-m")
            mod = cmdline.split()[idx + 1] if idx + 1 < len(cmdline.split()) else None
            if mod:
                return f"pkill -f {mod}   # or: kill {pid}"

    return f"kill {pid}"


# ── Public API ───────────────────────────────────────────────────────────────

def probe_ports(ports: list[int]) -> list[PortInfo]:
    """Probe each port and return a :class:`PortInfo` for each.

    The result list is in the same order as *ports*.  Always returns a list
    of the same length — one entry per port, whether free or busy.
    """
    results: list[PortInfo] = []
    for port in ports:
        free = _is_port_free(port)
        if free:
            results.append(PortInfo(port=port, is_free=True))
        else:
            pid, name, cmd = _find_port_owner(port)
            kill = _make_kill_suggestion(pid, name, cmd)
            results.append(PortInfo(
                port=port, is_free=False,
                pid=pid, process_name=name,
                cmdline=cmd, kill_suggestion=kill,
            ))
    return results


def check_ports(
    ports: list[int],
    *,
    label: str = "",
    verbose: bool = True,
) -> bool:
    """Check *ports* and print diagnostics for any that are occupied.

    Parameters
    ----------
    ports:
        List of TCP port numbers to check.
    label:
        Optional prefix printed before any warning (e.g. ``"[demo]"``).
    verbose:
        If *True* (default), print a tidy summary.  Pass *False* to suppress
        all output (useful in tests that just need the bool return value).

    Returns
    -------
    bool
        ``True`` if **all** ports are free, ``False`` if any are occupied.
    """
    infos = probe_ports(ports)
    busy = [i for i in infos if not i.is_free]

    if not busy:
        if verbose:
            tag = f"{label} " if label else ""
            print(f"  {tag}Ports {ports} are all free ✓")
        return True

    prefix = f"{label} " if label else ""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {prefix}WARNING — the following port(s) are already in use:", file=sys.stderr)
    for info in busy:
        print(_format_port_info(info), file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    return False


def assert_ports_free(ports: list[int], label: str = "") -> None:
    """Like :func:`check_ports` but raises :exc:`PortInUseError` on conflict.

    Use this in tests or strict startup paths where a busy port means the
    test/demo cannot proceed::

        assert_ports_free([10000, 8000], label="[demo]")

    Raises
    ------
    PortInUseError
        If any of *ports* is already bound.  The exception message contains
        full diagnostics and ready-to-paste kill commands.
    """
    infos = probe_ports(ports)
    busy = [i for i in infos if not i.is_free]
    if busy:
        # Print the same tidy output to stderr first
        prefix = f"{label} " if label else ""
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  {prefix}FATAL — port(s) are already in use:", file=sys.stderr)
        for info in busy:
            print(_format_port_info(info), file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        raise PortInUseError(busy)


def preflight_check(
    ports: dict[int, str],
    *,
    label: str = "",
    strict: bool = False,
) -> bool:
    """High-level pre-flight port check for demos and test fixtures.

    The single call to make at the top of any demo or test setup.
    Accepts a ``{port: description}`` dict so each port has a human-readable
    label in the output.

    Parameters
    ----------
    ports:
        Mapping of ``{port_number: human_readable_description}``, e.g.::

            {10000: "Deephaven server", 8000: "market data server"}
    label:
        Optional prefix for all messages (e.g. ``"[demo]"`` or ``"[test]"``).
    strict:
        If *False* (default, demo mode): print warnings but continue —
        the server itself will fail to bind if the port is truly blocked,
        giving a clear error at that point.
        If *True* (test mode): raise :exc:`PortInUseError` immediately so the
        test is skipped/failed before any server is started.

    Returns
    -------
    bool
        ``True`` if all ports are free, ``False`` if any are busy.
        In *strict* mode, ``False`` is never returned — a busy port raises.

    Examples
    --------
    Demo (soft)::

        from streaming.port_check import preflight_check
        preflight_check({10000: "Deephaven server", 8000: "market data server"})

    Test fixture (strict)::

        from streaming.port_check import preflight_check
        preflight_check({10000: "Deephaven server", 8000: "market data server"}, strict=True)
    """
    tag = f"{label} " if label else ""
    port_list = list(ports.keys())
    infos = probe_ports(port_list)
    busy = [i for i in infos if not i.is_free]

    if not busy:
        print(f"  {tag}Ports {port_list} are all free \u2713")
        return True

    # Print tidy diagnostics per busy port
    severity = "FATAL" if strict else "WARNING"
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {tag}{severity} — the following port(s) are already in use:",
          file=sys.stderr)
    for info in busy:
        desc = ports.get(info.port, "")
        desc_str = f" ({desc})" if desc else ""
        lines = [f"\n  Port {info.port}{desc_str}: IN USE"]
        if info.pid is not None:  # pid=0 = Docker sentinel
            lines.append(f"    Process : {info.process_name or 'unknown'} (PID {info.pid if info.pid else 'n/a'})")
        if info.cmdline:
            cmd = info.cmdline if len(info.cmdline) <= 120 else info.cmdline[:117] + "..."
            lines.append(f"    Command : {cmd}")
        if info.kill_suggestion:
            lines.append(f"    To free  : {info.kill_suggestion}")
        print("\n".join(lines) + "\n", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if strict:
        raise PortInUseError(busy)

    print("  Continuing — kill the process(es) above and re-run if startup fails.\n",
          file=sys.stderr)
    return False

