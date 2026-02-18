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
        self, X: list[str], separator=',', trunc=None, digits_per_timestamp=3, **kwargs
    ) -> np.ndarray:
        """Parse interleaved digit strings back to integer arrays."""
        width_used = self.metadata['width_used']

        def deinterleave_timestamp(interleaved_str):
            """Convert interleaved digits back to original values."""
            total_digits = len(interleaved_str)
            num_values = total_digits // width_used

            values = []
            for value_idx in range(num_values):
                value_digits = []
                for digit_pos in range(width_used):
                    pos = digit_pos * num_values + value_idx
                    if pos < total_digits:
                        value_digits.append(interleaved_str[pos])

                if value_digits:
                    values.append(int(''.join(value_digits)))

            return np.array(values)[:trunc] if trunc else np.array(values)

        result = np.array(
            [
                [
                    deinterleave_timestamp(timestamp)
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
