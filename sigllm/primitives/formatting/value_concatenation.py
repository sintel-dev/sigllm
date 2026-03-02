import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class ValueConcatenation(MultivariateFormattingMethod):
    """Formatting method that concatenates values directly."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('value_concatenation', verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=',', **kwargs) -> str:
        """Format array as string with concatenated values."""
        result = []
        for row in X:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(self, X: list[str], separator=',', trunc=None, num_dims=None, target_column=None, **kwargs) -> np.ndarray:
        """Extract values for the target dimension from each sample in each window as ints.
        
        Args:
            X (list[str]):
                list of strings, each string is a concatenation of num_dims values separated by separator
            separator (str):
                separator between values
            trunc (int): 
                Number of values to extract from each sample. If None, all values are extracted.
            num_dims (int): 
                Number of dimensions (mandatory if num_dims is not provided in config)
            target_column (int):
                Which dimension to extract (default 0). Can also be set via config.

        Returns:
            np.ndarray that holds int values for the target dimension for each sample in each window.
        """
        num_dims = num_dims or self.config.get("num_dims")
        if num_dims is None:
            raise ValueError("Cannot parse concatenated values without knowing the number of dimensions.")

        target_column = target_column if target_column is not None else self.config.get("target_column", 0)

        result = [
            [
                np.array(
                    [int(x) for x in entry.lstrip(separator).split(separator) if x]
                )[target_column::num_dims][:trunc]
                for entry in row
            ]
            for row in X
        ]
        out = np.array(result, dtype=object)
        return out
