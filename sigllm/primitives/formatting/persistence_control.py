import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod

class PersistenceControl(MultivariateFormattingMethod):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__("persistence_control", verbose=verbose, **kwargs)

    def format_as_string(self, data: np.ndarray, separator = ",") -> str:
        result = []
        for row in data[:, :, 0]:
            result.append(separator.join(map(str, row.flatten())))
        return result

    def format_as_integer(self, data: list[str], separator = ",", trunc = None) -> np.ndarray:
        result = [
            [np.array([int(x) for x in entry.lstrip(separator).split(separator) if x])[-1:]]
            for entry in data
        ]
        out = np.array(result, dtype=object)
        return out