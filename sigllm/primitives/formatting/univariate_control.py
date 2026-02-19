import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class UnivariateControl(MultivariateFormattingMethod):
    """Formatting method using univariate control strategy."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('univariate_control', verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=',', **kwargs) -> str:
        """Format array as string with univariate control."""
        result = []
        for row in X[:, :, 0]:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(self, X: list[str], separator=',', trunc=None, **kwargs) -> np.ndarray:
        """Parse string representation back to integer array."""
        result = [
            [
                np.array([int(x) for x in entry.lstrip(separator).split(separator) if x])[:trunc]
                for entry in row
            ]
            for row in X
        ]
        out = np.array(result, dtype=object)
        return out
