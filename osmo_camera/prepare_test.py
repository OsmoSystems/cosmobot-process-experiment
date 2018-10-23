from . import prepare as module


class TestHostname():
    def test_hostname_is_valid(self):
        assert module.hostname_is_valid('pi-cam-2222') is True

    def test_hostname_is_invalid(self):
        assert module.hostname_is_valid('I am a naughty hostname') is False


class TestGetExperimentVariants():
    def test_exposure_no_iso_uses_default_iso(self):
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

    def test_exposure_and_iso_generate_correct_variants(self):
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

    def test_exposure_and_iso_and_variant_generate_correct_variants(self):
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

    def test_only_variants_generate_correct_variants(self):
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

    def test_default_variants_generated(self):
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
