from . import prepare as module


def test_hostname_is_valid():
    assert module.hostname_is_valid('pi-cam-2222') is True


def test_hostname_is_invalid():
    assert module.hostname_is_valid('I am a naughty hostname') is False
