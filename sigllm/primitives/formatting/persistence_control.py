import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod

class PersistenceControl(MultivariateFormattingMethod):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__("persistence_control", verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator = ",", **kwargs) -> str:
        result = []
        for row in X[:, :, 0]:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(self, X: list[str], separator = ",", trunc = None, **kwargs) -> np.ndarray:
        result = [
            [np.array([int(x) for x in entry.lstrip(separator).split(separator) if x])[-1:]]
            for entry in X
        ]
        out = np.array(result, dtype=object)
        return out
