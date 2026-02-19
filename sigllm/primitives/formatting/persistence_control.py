import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class PersistenceControl(MultivariateFormattingMethod):
    """Formatting method using persistence control strategy."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('persistence_control', verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=',', **kwargs) -> str:
        """Format array as string with persistence control."""
        result = []
        for row in X[:, :, 0]:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(self, X: list[str], separator=',', trunc=None, **kwargs) -> np.ndarray:
        """Parse string representation back to integer array."""
        result = [
            [np.array([int(x) for x in entry.lstrip(separator).split(separator) if x])[-1:]]
            for entry in X
        ]
        out = np.array(result, dtype=object)
        return out
