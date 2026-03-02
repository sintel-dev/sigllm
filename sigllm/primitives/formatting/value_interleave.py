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
        self, X: list[str], separator=',', trunc=None, digits_per_timestamp=3, target_column=None, **kwargs
    ) -> np.ndarray:
        """Parse interleaved value strings back to integer arrays for the target dimension.

        Args:
            X (list[str]):
                list of strings, each string is a concatenation of 
                num_dims values separated by separator.
            separator (str):
                separator between values
            trunc (int): 
                Number of values to extract from each sample. If None, all values are extracted.
            digits_per_timestamp (int):
                Number of digits to extract from each timestamp.
            target_column (int):
                Which dimension to extract (default 0). Can also be set via config.

        Returns:
            np.ndarray that holds int values for the target dimension for each sample in each window.
        """
        width_used = self.metadata['width_used']
        target_column = target_column if target_column is not None else self.config.get('target_column', 0)

        def parse_target_column_from_timestamp(timestamp):
            arr = [
                int(timestamp[i : i + width_used]) for i in range(0, len(timestamp), width_used)
            ]
            if target_column < len(arr):
                return arr[target_column]
            return None

        result = []
        for entry in X:
            row = []
            for sample in entry:
                parts = (
                    sample.lstrip(separator)
                    .rstrip(separator)
                    .split(separator)
                )
                vals = np.array([parse_target_column_from_timestamp(ts) for ts in parts if ts])
                if trunc is not None:
                    vals = vals[:trunc]
                row.append(vals)
            result.append(row)
        return np.array(result, dtype=object)
