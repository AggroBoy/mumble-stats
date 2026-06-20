#!/usr/bin/env python3
"""Tests for the mumble-users command."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import socket
import struct
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "mumble-users"


def load_module():
    loader = importlib.machinery.SourceFileLoader("mumble_users", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


mumble_users = load_module()


class ParseHostPortTests(unittest.TestCase):
    def test_host_uses_default_port(self):
        self.assertEqual(
            mumble_users.parse_host_port("example.com", 64738),
            ("example.com", 64738),
        )

    def test_host_colon_port(self):
        self.assertEqual(
            mumble_users.parse_host_port("example.com:12345", 64738),
            ("example.com", 12345),
        )

    def test_ipv4_colon_port(self):
        self.assertEqual(
            mumble_users.parse_host_port("127.0.0.1:12345", 64738),
            ("127.0.0.1", 12345),
        )

    def test_unbracketed_ipv6_uses_default_port(self):
        self.assertEqual(
            mumble_users.parse_host_port("::1", 64738),
            ("::1", 64738),
        )

    def test_bracketed_ipv6_colon_port(self):
        self.assertEqual(
            mumble_users.parse_host_port("[::1]:12345", 64738),
            ("::1", 12345),
        )

    def test_invalid_port_is_rejected(self):
        with self.assertRaises(ValueError):
            mumble_users.parse_host_port("example.com:70000", 64738)

    def test_env_server_uses_default_port(self):
        self.assertEqual(
            mumble_users.resolve_target(None, None, {"MUMBLE_SERVER": "example.com"}),
            ("example.com", 64738),
        )

    def test_env_server_uses_env_port(self):
        self.assertEqual(
            mumble_users.resolve_target(
                None,
                None,
                {"MUMBLE_SERVER": "example.com", "MUMBLE_SERVER_PORT": "12345"},
            ),
            ("example.com", 12345),
        )

    def test_env_server_embedded_port_overrides_env_port(self):
        self.assertEqual(
            mumble_users.resolve_target(
                None,
                None,
                {"MUMBLE_SERVER": "example.com:23456", "MUMBLE_SERVER_PORT": "12345"},
            ),
            ("example.com", 23456),
        )

    def test_cli_values_override_env_values(self):
        self.assertEqual(
            mumble_users.resolve_target(
                "cli.example.com",
                34567,
                {"MUMBLE_SERVER": "env.example.com", "MUMBLE_SERVER_PORT": "12345"},
            ),
            ("cli.example.com", 34567),
        )

    def test_cli_server_embedded_port_overrides_cli_port(self):
        self.assertEqual(
            mumble_users.resolve_target("cli.example.com:45678", 34567, {}),
            ("cli.example.com", 45678),
        )

    def test_missing_server_is_rejected(self):
        with self.assertRaises(ValueError):
            mumble_users.resolve_target(None, None, {})

    def test_invalid_env_port_is_rejected(self):
        with self.assertRaises(ValueError):
            mumble_users.resolve_target(
                None,
                None,
                {"MUMBLE_SERVER": "example.com", "MUMBLE_SERVER_PORT": "not-a-port"},
            )


class PacketTests(unittest.TestCase):
    def test_format_server_brackets_ipv6(self):
        self.assertEqual(mumble_users.format_server("::1", 64738), "[::1]:64738")

    def test_format_version_decodes_mumble_version(self):
        self.assertEqual(mumble_users.format_version(0x00010502), "1.5.2")

    def test_format_user_count_uses_singular_for_one_user(self):
        self.assertEqual(mumble_users.format_user_count(1), "1 user")
        self.assertEqual(mumble_users.format_user_count(2), "2 users")

    def test_request_is_legacy_extended_ping(self):
        self.assertEqual(
            mumble_users.build_ping_request(0x0102030405060708),
            b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08",
        )

    def test_parse_response_extracts_user_count(self):
        response = struct.pack(">IQIII", 0x01020304, 0x10, 7, 100, 72000)

        stats = mumble_users.parse_ping_response(response, expected_nonce=0x10)

        self.assertEqual(stats.version, 0x01020304)
        self.assertEqual(stats.users, 7)
        self.assertEqual(stats.max_users, 100)
        self.assertEqual(stats.max_bandwidth, 72000)

    def test_parse_response_rejects_short_packets(self):
        with self.assertRaises(mumble_users.MumbleQueryError):
            mumble_users.parse_ping_response(b"short")

    def test_parse_response_rejects_nonce_mismatch(self):
        response = struct.pack(">IQIII", 0x01020304, 0x10, 7, 100, 72000)
        with self.assertRaises(mumble_users.MumbleQueryError):
            mumble_users.parse_ping_response(response, expected_nonce=0x11)


class QueryTests(unittest.TestCase):
    def run_fake_server_query(self, argv):
        ready = threading.Event()
        seen_request = []

        def server():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", 0))
                seen_request.append(sock.getsockname())
                ready.set()
                data, client = sock.recvfrom(1024)
                _version, nonce = struct.unpack(">IQ", data)
                sock.sendto(struct.pack(">IQIII", 0x00010502, nonce, 42, 100, 72000), client)

        thread = threading.Thread(target=server)
        thread.start()
        self.assertTrue(ready.wait(2))
        _host, port = seen_request[0]

        output = io.StringIO()
        with redirect_stdout(output):
            status = mumble_users.main([*argv, "--port", str(port), "--timeout", "1"])

        thread.join(2)
        return status, output.getvalue(), port

    def test_main_prints_user_count_from_udp_response(self):
        status, output, _port = self.run_fake_server_query(["127.0.0.1"])

        self.assertEqual(status, 0)
        self.assertEqual(output, "42\n")

    def test_verbose_main_prints_human_readable_server_stats(self):
        status, output, port = self.run_fake_server_query(["127.0.0.1", "--verbose"])

        self.assertEqual(status, 0)
        self.assertEqual(
            output,
            f"127.0.0.1:{port} (version: 1.5.2): 42 users online (maximum: 100)\n",
        )


class MainEnvTests(unittest.TestCase):
    def test_main_uses_env_server_and_env_port_when_server_argument_is_omitted(self):
        stats = mumble_users.ServerStats(
            version=0x00010502,
            users=3,
            max_users=100,
            max_bandwidth=72000,
        )

        output = io.StringIO()
        with mock.patch.dict(
            mumble_users.os.environ,
            {"MUMBLE_SERVER": "env.example.com", "MUMBLE_SERVER_PORT": "12345"},
            clear=True,
        ):
            with mock.patch.object(mumble_users, "query_server", return_value=stats) as query:
                with redirect_stdout(output):
                    status = mumble_users.main([])

        self.assertEqual(status, 0)
        query.assert_called_once_with("env.example.com", 12345, 2.0)
        self.assertEqual(output.getvalue(), "3\n")


if __name__ == "__main__":
    unittest.main()
