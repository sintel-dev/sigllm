import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class UnivariateControl(MultivariateFormattingMethod):
    """Formatting method using univariate control strategy."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('univariate_control', verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=',', target_column=None, **kwargs) -> str:
        """Format array as string with univariate control.

        Args:
            X (np.ndarray):
                Input array with shape (num_windows, num_timestamps, num_dims).
            separator (str):
                Separator between values.
            target_column (int):
                Which dimension to encode (default 0). Can also be set via config.

        Returns:
            list[str]:
                List of strings, one per window, containing only
                the target dimension values.
        """
        if target_column is None:
            target_column = self.config.get('target_column', 0)
        result = []
        for row in X[:, :, target_column]:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(
        self, X: list[str], separator=',', trunc=None, target_column=None, **kwargs
    ) -> np.ndarray:
        """Parse string representation back to integer array.

        Args:
            X (list[str]):
                List of strings to parse.
            separator (str):
                Separator between values.
            trunc (int):
                Number of values to extract. If None, all values are extracted.
            target_column (int):
                Accepted for API consistency (default 0). The string already contains
                only the target dimension, so this parameter has no effect on parsing.

        Returns:
            np.ndarray that holds int values for each sample in each window.
        """
        result = [
            [
                np.array([int(x) for x in entry.lstrip(separator).split(separator) if x])[:trunc]
                for entry in row
            ]
            for row in X
        ]
        out = np.array(result, dtype=object)
        return out
