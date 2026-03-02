import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class DigitInterleave(MultivariateFormattingMethod):
    """Formatting method that interleaves digits from multiple values."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('digit_interleave', verbose=verbose, **kwargs)

    def format_as_string(
        self, X: np.ndarray, digits_per_timestamp=3, separator=',', **kwargs
    ) -> str:
        """Format array as string with interleaved digits."""
        max_digits = max(len(str(abs(int(v)))) for window in X for ts in window for v in ts)
        width_used = max(digits_per_timestamp, max_digits)
        self.metadata['width_used'] = width_used

        def interleave_digits(timestamp):
            str_values = [str(int(val)) for val in timestamp]
            padded_values = [s.zfill(width_used) for s in str_values]
            result_str = ''
            for digit_pos in range(width_used):
                for padded_val in padded_values:
                    result_str += padded_val[digit_pos]

            return result_str

        result = [
            separator.join(interleave_digits(timestamp) for timestamp in window) + separator
            for window in X
        ]
        return result

    def format_as_integer(
        self,
        X: list[str],
        separator=',',
        trunc=None,
        digits_per_timestamp=3,
        target_column=None,
        **kwargs,
    ) -> np.ndarray:
        """Parse interleaved digit strings back to integer arrays for the target column.

        Args:
            X (list[str]):
                list of strings, each string is a concatenation of
                interleaved digit values separated by separator.
            separator (str):
                separator between values
            trunc (int):
                Number of timestamps to extract from each sample.
                If None, all timestamps are extracted.
            digits_per_timestamp (int):
                Number of digits to extract from each timestamp.
            target_column (int):
                Which column to extract (default 0). Can also be set via config.

        Returns:
            np.ndarray that holds int values for the target column for each sample in each window.
        """
        width_used = self.metadata['width_used']
        if target_column is None:
            target_column = self.config.get('target_column', 0)

        def deinterleave_timestamp_target_column(interleaved_str):
            """Convert interleaved digits back to original values and extract target dimension."""
            total_digits = len(interleaved_str)
            num_values = total_digits // width_used

            if target_column >= num_values:
                return np.array([None])

            value_digits = []
            for digit_pos in range(width_used):
                pos = digit_pos * num_values + target_column
                if pos < total_digits:
                    value_digits.append(interleaved_str[pos])

            if value_digits:
                return np.array([int(''.join(value_digits))])
            return np.array([None])

        result = np.array(
            [
                [
                    deinterleave_timestamp_target_column(timestamp)
                    for sample in entry
                    for timestamp in sample
                    .lstrip(separator)
                    .rstrip(separator)
                    .split(separator)[:trunc]
                    if timestamp.strip()
                ]
                for entry in X
            ],
            dtype=object,
        )
        return result
