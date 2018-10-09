from . import prepare as module


def test_is_hostname_valid():
    assert module.is_hostname_valid('pi-cam-2222') is True


def test_is_hostname_invalid():
    assert module.is_hostname_valid('I am a naughty hostname') is False
