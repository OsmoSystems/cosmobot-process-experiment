from . import prepare as module


def test_hostname_is_valid():
    assert module.hostname_is_valid('pi-cam-2222') is True


def test_hostname_is_invalid():
    assert module.hostname_is_valid('I am a naughty hostname') is False


def test_get_experiment_variants_exposure_no_iso():
    args = {
        'name': 'test',
        'interval': 10,
        'variant': [],
        'exposures': [100, 200],
        'isos': None
    }

    expected = [
        module.ExperimentVariant(capture_params='" -ss 100 -ISO 100"'),
        module.ExperimentVariant(capture_params='" -ss 200 -ISO 100"')
    ]

    actual = module.get_experiment_variants(args)
    assert actual == expected


def test_get_experiment_variants_exposure_and_iso():
    args = {
        'name': 'test',
        'interval': 10,
        'variant': [],
        'exposures': [100, 200],
        'isos': [100, 200]
    }

    expected = [
        module.ExperimentVariant(capture_params='" -ss 100 -ISO 100"'),
        module.ExperimentVariant(capture_params='" -ss 100 -ISO 200"'),
        module.ExperimentVariant(capture_params='" -ss 200 -ISO 100"'),
        module.ExperimentVariant(capture_params='" -ss 200 -ISO 200"')
    ]

    actual = module.get_experiment_variants(args)
    assert actual == expected


def test_get_experiment_variants_exposure_iso_and_variant():
    args = {
        'name': 'test',
        'interval': 10,
        'variant': [' -ss 4000000 -ISO 100'],
        'exposures': [100, 200],
        'isos': [100, 200]
    }

    expected = [
        module.ExperimentVariant(capture_params=' -ss 4000000 -ISO 100'),
        module.ExperimentVariant(capture_params='" -ss 100 -ISO 100"'),
        module.ExperimentVariant(capture_params='" -ss 100 -ISO 200"'),
        module.ExperimentVariant(capture_params='" -ss 200 -ISO 100"'),
        module.ExperimentVariant(capture_params='" -ss 200 -ISO 200"')
    ]

    actual = module.get_experiment_variants(args)
    assert actual == expected


def test_get_experiment_variants_only_variants():
    args = {
        'name': 'test',
        'interval': 10,
        'variant': [' -ss 1000000 -ISO 100', ' -ss 1100000 -ISO 100'],
        'exposures': None,
        'isos': None
    }

    expected = [
        module.ExperimentVariant(capture_params=' -ss 1000000 -ISO 100'),
        module.ExperimentVariant(capture_params=' -ss 1100000 -ISO 100')
    ]

    actual = module.get_experiment_variants(args)
    assert actual == expected


def test_get_experiment_variants_nothing_specified():
    args = {
        'name': 'test',
        'interval': 10,
        'variant': [],
        'exposures': None,
        'isos': None
    }

    expected = [
        module.ExperimentVariant(capture_params=' -ss 1500000 -ISO 100')
    ]

    actual = module.get_experiment_variants(args)
    assert actual == expected
