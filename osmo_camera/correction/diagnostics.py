import numpy as np
import pandas as pd

import warnings


class CorrectionWarning(Warning):
    ''' Warning type used when something fishy is detected in our correction steps
    to use:

    >>> import warnings
    >>> warnings.warn('message', category=CorrectionWarning)

    By default, only one identical warning will be shown originating on a particular line,
    and subsequent identical warnings will be ignored.
    This can be controlled using the warnings API documented at https://docs.python.org/3/library/warnings.html
    '''


def warn_if_any_true(possible_warnings_series):
    ''' Raise a CorrectionWarning if any of the values in a boolean series are True.

    Args:
        possible_warnings_series: pandas Series of booleans, which if True, should raise a warning.
    Returns:
        None
    Warns:
        uses the warnings API and CorrectionWarning if any red flags are present.
    '''
    if possible_warnings_series.dtype != np.bool_:
        raise ValueError(
            f'possible warnings {possible_warnings_series} must be a boolean series'
        )

    # Filter only to True values (`possible_warnings_series` itself is a boolean series which can be used to filter)
    raised_warnings = possible_warnings_series[possible_warnings_series]
    if raised_warnings.any():
        warnings.warn(
            f'Diagnostic warning flags: {raised_warnings.index.values}',
            category=CorrectionWarning,
            stacklevel=2  # Raise on the immediate caller's level, not this level
        )


def run_diagnostics(images_series_before, images_series_after, diagnostics_fn):
    ''' Run a diagnostics function on a pair of series of images
    Note: the diagnostic function may raise warnings.

    Args:
        image_series_before: Series of images, before transformation
        image_series_after: Series of images, after a particular transformation
        diagnostics_fn: the diagnostics function designed to be paired with the transformation
            This function should take RGB images "before" and "after" and the raw image path image_path.
    Returns:
        DataFrame with columns of diagnostic values

    '''
    return pd.DataFrame.from_dict(
        {
            raw_image_path: diagnostics_fn(
                before=images_series_before[raw_image_path],
                after=images_series_after[raw_image_path],
                image_path=raw_image_path,
            )
            for raw_image_path in images_series_before.index
        },
        orient='index'  # file paths should end up as the dataframe index (default would make them column names)
    )