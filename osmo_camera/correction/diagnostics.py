import numpy as np

import warnings


class CorrectionWarning(Warning):
    """ Warning type used when something fishy is detected in our correction steps
    to use:

    >>> import warnings
    >>> warnings.warn('message', category=CorrectionWarning)

    By default, only one identical warning will be shown originating on a particular line,
    and subsequent identical warnings will be ignored.
    This can be controlled using the warnings API documented at https://docs.python.org/3/library/warnings.html
    """


def warn_if_any_true(possible_warnings_series):
    """ Raise a CorrectionWarning if any of the values in a boolean series are True.

    Args:
        possible_warnings_series: pandas Series of booleans, which if True, should raise a warning.
    Returns:
        None
    Warns:
        uses the warnings API and CorrectionWarning if any red flags are present.
    """
    if possible_warnings_series.dtype != np.bool_:
        raise ValueError(
            f"possible warnings {possible_warnings_series} must be a boolean series"
        )

    # Filter only to True values (`possible_warnings_series` itself is a boolean series which can be used to filter)
    raised_warnings = possible_warnings_series[possible_warnings_series]
    if raised_warnings.any():
        warnings.warn(
            f"Diagnostic warning flags: {raised_warnings.index.values}",
            category=CorrectionWarning,
            stacklevel=2,  # Raise on the immediate caller's level, not this level
        )
