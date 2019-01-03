import warnings

import pandas as pd

import osmo_camera.correction.diagnostics as module


class TestWarnIfAnyTrue:
    def test_raises_warning_with_true_indexes_only(self, mocker):
        mock_warn = mocker.patch.object(module.warnings, 'warn')

        module.warn_if_any_true(pd.Series({
            'bad_thing': True,
            'worse_thing': True,
            'total_disaster': False,
        }))

        actual_warning_message = mock_warn.call_args[0][0]

        assert 'bad_thing' in actual_warning_message
        assert 'worse_thing' in actual_warning_message
        assert 'total_disaster' not in actual_warning_message

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


class TestRunDiagnostics:
    def test_diagnostics_run_per_image(self, mocker):
        diagnostics_fn = mocker.Mock()
        sentinel = mocker.sentinel

        image_series_before = pd.Series({
            sentinel.path_one: sentinel.image_one_before,
            sentinel.path_two: sentinel.image_two_before,
        })
        image_series_after = pd.Series({
            sentinel.path_one: sentinel.image_one_after,
            sentinel.path_two: sentinel.image_two_after,
        })
        module.run_diagnostics(
            image_series_before,
            image_series_after,
            diagnostics_fn
        )

        diagnostics_fn.assert_has_calls([
            mocker.call(
                before=sentinel.image_one_before,
                after=sentinel.image_one_after,
                image_path=sentinel.path_one,
            ),
            mocker.call(
                before=sentinel.image_two_before,
                after=sentinel.image_two_after,
                image_path=sentinel.path_two,
            )
        ])

    def test_diagnostics_returned_as_df(self, mocker):
        diagnostics = pd.Series({'this_one_thing_is_broken': True})
        diagnostics_fn = mocker.Mock()
        diagnostics_fn.return_value = diagnostics
        sentinel = mocker.sentinel

        image_series_before = pd.Series({sentinel.path_one: sentinel.image_one_before})
        image_series_after = pd.Series({sentinel.path_one: sentinel.image_one_after})

        expected = pd.DataFrame.from_dict(
            {
                sentinel.path_one: diagnostics
            },
            # Be explicit here that the path is the index, not the column name
            orient='index'
        )
        print(expected)
        print(expected.columns)

        actual = module.run_diagnostics(
            image_series_before,
            image_series_after,
            diagnostics_fn
        )

        pd.testing.assert_frame_equal(actual, expected)
