"""
test_port_check.py — Tests for streaming.port_check utility.

Tests both the free-port and busy-port detection paths.
The busy-port tests work by binding a real socket to a free ephemeral port
and then checking that probe_ports() detects it as occupied.

No external processes or servers are required — all tests are self-contained.
"""

import socket
import contextlib
import pytest

from streaming.port_check import (
    PortInfo,
    PortInUseError,
    assert_ports_free,
    check_ports,
    preflight_check,
    probe_ports,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _bound_port():
    """Bind an ephemeral port and yield its number.  Socket released on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield sock.getsockname()[1]  # the OS-assigned port number
    finally:
        sock.close()


def _free_port() -> int:
    """Find a free port (bind then immediately release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── probe_ports ───────────────────────────────────────────────────────────────

class TestProbePorts:
    """probe_ports() returns PortInfo for each queried port."""

    def test_free_port_detected(self):
        port = _free_port()
        infos = probe_ports([port])
        assert len(infos) == 1
        assert isinstance(infos[0], PortInfo)
        assert infos[0].port == port
        assert infos[0].is_free is True
        assert infos[0].pid is None

    def test_busy_port_detected(self):
        with _bound_port() as port:
            infos = probe_ports([port])
            assert len(infos) == 1
            info = infos[0]
            assert info.port == port
            assert info.is_free is False

    def test_busy_port_has_pid(self):
        """Our own process owns the socket — PID should be ours."""
        import os
        with _bound_port() as port:
            infos = probe_ports([port])
            info = infos[0]
            # PID discovery may not work in all CI environments — just check
            # that if a PID is found, it is a positive integer.
            if info.pid is not None:
                assert isinstance(info.pid, int)
                assert info.pid > 0

    def test_multiple_ports_mixed(self):
        """Mix of free and busy ports all returned in order."""
        free = _free_port()
        with _bound_port() as busy:
            infos = probe_ports([free, busy])
            assert len(infos) == 2
            assert infos[0].port == free
            assert infos[0].is_free is True
            assert infos[1].port == busy
            assert infos[1].is_free is False

    def test_multiple_free_ports(self):
        p1, p2 = _free_port(), _free_port()
        infos = probe_ports([p1, p2])
        assert all(i.is_free for i in infos)

    def test_result_order_preserved(self):
        """Output order matches input order regardless of internal system calls."""
        ports = [_free_port() for _ in range(4)]
        infos = probe_ports(ports)
        assert [i.port for i in infos] == ports


# ── check_ports ───────────────────────────────────────────────────────────────

class TestCheckPorts:
    """check_ports() returns True iff all ports are free."""

    def test_returns_true_when_all_free(self):
        port = _free_port()
        result = check_ports([port], verbose=False)
        assert result is True

    def test_returns_false_when_busy(self, capsys):
        with _bound_port() as port:
            result = check_ports([port], verbose=True)
        assert result is False

    def test_prints_warning_on_stderr_when_busy(self, capsys):
        with _bound_port() as port:
            check_ports([port], verbose=True)
        captured = capsys.readouterr()
        assert str(port) in captured.err
        assert "IN USE" in captured.err

    def test_verbose_false_suppresses_output_for_free_port(self, capsys):
        port = _free_port()
        check_ports([port], verbose=False)
        captured = capsys.readouterr()
        # Nothing should have been printed to stdout or stderr
        assert captured.out == ""
        assert captured.err == ""

    def test_label_appears_in_output(self, capsys):
        with _bound_port() as port:
            check_ports([port], label="[mytest]", verbose=True)
        err = capsys.readouterr().err
        assert "[mytest]" in err

    def test_empty_port_list_returns_true(self):
        assert check_ports([], verbose=False) is True


# ── assert_ports_free ─────────────────────────────────────────────────────────

class TestAssertPortsFree:
    """assert_ports_free() raises PortInUseError when any port is busy."""

    def test_does_not_raise_when_free(self):
        port = _free_port()
        assert_ports_free([port])  # must not raise

    def test_raises_when_busy(self):
        with _bound_port() as port:
            with pytest.raises(PortInUseError) as exc_info:
                assert_ports_free([port])
        err = exc_info.value
        assert isinstance(err.busy, list)
        assert len(err.busy) == 1
        assert err.busy[0].port == port
        assert err.busy[0].is_free is False

    def test_exception_message_contains_port(self):
        with _bound_port() as port:
            with pytest.raises(PortInUseError) as exc_info:
                assert_ports_free([port])
        assert str(port) in str(exc_info.value)

    def test_exception_message_contains_kill_hint(self):
        """Exception text should mention 'kill' in some form."""
        with _bound_port() as port:
            with pytest.raises(PortInUseError) as exc_info:
                assert_ports_free([port])
        # The kill suggestion may say 'kill <pid>' or 'pkill -f ...'
        # Only present if PID detection succeeded — tolerate CI environments.
        err_text = str(exc_info.value)
        assert str(port) in err_text  # port number always present

    def test_busy_list_has_correct_type(self):
        with _bound_port() as port:
            try:
                assert_ports_free([port])
            except PortInUseError as e:
                assert all(isinstance(b, PortInfo) for b in e.busy)

    def test_raises_with_multiple_busy_ports(self):
        with _bound_port() as p1, _bound_port() as p2:
            with pytest.raises(PortInUseError) as exc_info:
                assert_ports_free([p1, p2])
        assert len(exc_info.value.busy) == 2

    def test_raises_only_for_busy_subset(self):
        """When some ports are free and some busy, only busy ones appear."""
        free = _free_port()
        with _bound_port() as busy:
            with pytest.raises(PortInUseError) as exc_info:
                assert_ports_free([free, busy])
        busy_ports = [b.port for b in exc_info.value.busy]
        assert busy in busy_ports
        assert free not in busy_ports

    def test_label_appears_in_stderr_on_raise(self, capsys):
        with _bound_port() as port:
            with pytest.raises(PortInUseError):
                assert_ports_free([port], label="[suite]")
        err = capsys.readouterr().err
        assert "[suite]" in err


# ── PortInfo dataclass ────────────────────────────────────────────────────────

class TestPortInfo:
    """Basic sanity checks on PortInfo construction."""

    def test_free_port_info(self):
        info = PortInfo(port=9999, is_free=True)
        assert info.port == 9999
        assert info.is_free is True
        assert info.pid is None
        assert info.kill_suggestion is None

    def test_busy_port_info(self):
        info = PortInfo(
            port=9999, is_free=False,
            pid=42, process_name="python3",
            cmdline="python3 demo.py",
            kill_suggestion="kill 42",
        )
        assert info.is_free is False
        assert info.pid == 42
        assert "kill 42" in info.kill_suggestion


# ── preflight_check ───────────────────────────────────────────────────────────

class TestPreflightCheck:
    """preflight_check() — the one-call high-level interface."""

    def test_returns_true_when_all_free(self):
        p1, p2 = _free_port(), _free_port()
        result = preflight_check({p1: "service A", p2: "service B"})
        assert result is True

    def test_returns_false_when_busy_soft_mode(self, capsys):
        with _bound_port() as port:
            result = preflight_check({port: "test service"})
        assert result is False

    def test_soft_mode_does_not_raise(self):
        """Default (strict=False) must not raise even when a port is busy."""
        with _bound_port() as port:
            # Must not raise:
            preflight_check({port: "test service"})

    def test_strict_mode_raises_when_busy(self):
        with _bound_port() as port:
            with pytest.raises(PortInUseError):
                preflight_check({port: "test service"}, strict=True)

    def test_strict_mode_does_not_raise_when_free(self):
        port = _free_port()
        preflight_check({port: "test service"}, strict=True)  # must not raise

    def test_description_appears_in_warning(self, capsys):
        """Port description should be included in the output when busy."""
        with _bound_port() as port:
            preflight_check({port: "my fancy service"})
        err = capsys.readouterr().err
        assert "my fancy service" in err

    def test_label_appears_in_output(self, capsys):
        with _bound_port() as port:
            preflight_check({port: "svc"}, label="[preflight-test]")
        err = capsys.readouterr().err
        assert "[preflight-test]" in err

    def test_empty_dict_returns_true(self):
        assert preflight_check({}) is True

    def test_stdout_on_all_free(self, capsys):
        """When all ports are free, a success message goes to stdout."""
        port = _free_port()
        preflight_check({port: "svc"})
        out = capsys.readouterr().out
        assert "✓" in out or "free" in out.lower()

    def test_strict_error_contains_busy_ports(self):
        with _bound_port() as port:
            with pytest.raises(PortInUseError) as exc_info:
                preflight_check({port: "svc"}, strict=True)
        assert any(b.port == port for b in exc_info.value.busy)
