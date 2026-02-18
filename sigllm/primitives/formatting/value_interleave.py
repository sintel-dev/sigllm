import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class ValueInterleave(MultivariateFormattingMethod):
    """Formatting method that interleaves values from multiple dimensions."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('value_interleave', verbose=verbose, **kwargs)

    def format_as_string(
        self, X: np.ndarray, digits_per_timestamp=3, separator=',', **kwargs
    ) -> str:
        """Format array as string with interleaved values."""
        max_digits = max(len(str(abs(int(v)))) for window in X for ts in window for v in ts)
        width_used = max(digits_per_timestamp, max_digits)
        self.metadata['width_used'] = width_used
        result = [
            separator.join(
                ''.join(str(int(val)).zfill(width_used)[:width_used] for val in timestamp)
                for timestamp in window
            )
            + separator
            for window in X
        ]
        return result

    def format_as_integer(
        self, X: list[str], separator=',', trunc=None, digits_per_timestamp=3, **kwargs
    ) -> np.ndarray:
        """Parse interleaved value strings back to integer arrays."""
        width_used = self.metadata['width_used']

        def parse_timestamp(timestamp):
            return np.array([
                int(timestamp[i : i + width_used]) for i in range(0, len(timestamp), width_used)
            ])[:trunc]

        result = np.array(
            [
                [
                    parse_timestamp(timestamp)
                    for sample in entry
                    for timestamp in sample
                    .lstrip(separator)
                    .rstrip(separator)
                    .split(separator)[:trunc]
                ]
                for entry in X
            ],
            dtype=object,
        )

        if result.ndim == 2:
            result = np.expand_dims(result, axis=-1)
        return result
