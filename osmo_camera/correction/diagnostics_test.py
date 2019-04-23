import warnings

import pandas as pd
import pytest

import osmo_camera.correction.diagnostics as module


class TestWarnIfAnyTrue:
    def test_raises_warning_with_true_indexes_only(self, mocker):
        mock_warn = mocker.patch.object(module.warnings, 'warn')

        module.warn_if_any_true(pd.Series({
            'bad_thing': True,
            'worse_thing': True,
            'total_disaster': False,
        }))

        actual_warning_message = mock_warn.call_args[0][0]  # indexing: first argument from first call

        assert 'bad_thing' in actual_warning_message
        assert 'worse_thing' in actual_warning_message
        assert 'total_disaster' not in actual_warning_message

    def test_blows_up_on_non_boolean_series(self, mocker):
        with pytest.raises(ValueError):
            module.warn_if_any_true(pd.Series({'that\'s not a knife': 'this is a knife'}))

    def test_no_warning_if_all_falsey(self, mocker):
        mock_warn = mocker.patch.object(module.warnings, 'warn')

        module.warn_if_any_true(pd.Series({'everything blew up': False}))

        mock_warn.assert_not_called()

    def test_raised_warning_doesnt_explode(self, mocker):
        # Since the other tests in this suite mock out warnings.warn,
        # this smoke test just makes sure things are wired up OK

        with warnings.catch_warnings():
            # Silence warnings so they don't end up at the top level
            warnings.simplefilter("ignore")
            module.warn_if_any_true(pd.Series({'broken': True}))
