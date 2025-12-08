from .multivariate_formatting import MultivariateFormattingMethod
import numpy as np
import re

class JSONFormat(MultivariateFormattingMethod):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__("json_format", verbose=verbose, **kwargs)

    def format_as_string(self, data: np.ndarray, separator = ",") -> str:
        def window_to_json(data):
            rows = []
            for row in data:
                parts = [f"d{i}:{val}" for i, val in enumerate(row)]
                rows.append(",".join(parts))
            return ",".join(rows)
        
        out = [window_to_json(window) for window in data]
        return out


    def format_as_integer(self, data, trunc=None):
        batch_rows = []
        for window in data:
            samples = []
            for sample in window:
                tokens = re.findall(r'd\d+:\d+', sample)
                flat, current = [], []
                for token in tokens:
                    key, val = token.split(":")
                    if key == "d0" and current:
                        flat.extend(current)
                        current = []
                    current.append(int(val))
                if current:
                    flat.extend(current)
                if trunc:
                    flat = flat[:trunc]
                samples.append(flat)
            batch_rows.append(samples)
        return np.array(batch_rows, dtype=object)




if __name__ == "__main__":
    method = JSONFormat()
    method.test_multivariate_formatting_validity(verbose=False)
    method.run_pipeline(multivariate_allowed_symbols=["d", ":", ","])