"""Every test is offline: no socket connection or DNS lookup is permitted."""
import socket

import pytest


@pytest.fixture(autouse=True, scope='session')
def no_network():
    def blocked(*args, **kwargs):
        raise AssertionError('Tests do not permit network access')
    patch = pytest.MonkeyPatch()
    for target, name in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                         (socket, 'create_connection'), (socket, 'getaddrinfo')):
        patch.setattr(target, name, blocked)
    yield
    patch.undo()
