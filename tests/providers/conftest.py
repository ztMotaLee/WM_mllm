"""Provider tests are hermetic and permit no sockets or DNS lookups."""
from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Network access is disabled in provider tests")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, reject)
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)
