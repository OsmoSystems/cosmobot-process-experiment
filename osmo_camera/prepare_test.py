import pytest

from . import prepare as module


@pytest.mark.parametrize("hostname,expected_is_valid", [
    ('pi-cam-CF22', True),
    ('pi-cam-4321', True),
    ('sneaky-pi-cam-CF22', False),
    ('pi-cam-12345', False),
    ('pi-cam-1234-and-more', False),
    ('I am a naughty hostname', False),
])
def test_hostname_is_valid(hostname, expected_is_valid):
    assert module.hostname_is_valid(hostname) == expected_is_valid
