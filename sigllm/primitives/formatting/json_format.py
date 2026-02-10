import re

import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class JSONFormat(MultivariateFormattingMethod):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__("json_format", verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=",", **kwargs) -> str:
        def window_to_json(X):
            rows = []
            for row in X:
                parts = [f"d{i}:{val}" for i, val in enumerate(row)]
                rows.append(",".join(parts))
            return ",".join(rows)
        
        out = [window_to_json(window) for window in X]
        return out

    def format_as_integer(self, X, trunc=None, steps_ahead=None, **kwargs):
        """
        Parse model output and extract d0 values for specified steps ahead.
        
        Args:
            X: Model output containing tokens like "d0:1,d1:2,d0:3,d1:4..."
            trunc: Legacy parameter for truncation (used when steps_ahead is None)
            steps_ahead: List of step indices to extract (e.g., [1,3,5,10])
                         If None, uses legacy behavior with trunc parameter.
        
        Returns:
            If steps_ahead is None: np.array of shape (batch, samples) with truncated flat values
            If steps_ahead is provided: dict mapping step -> np.array of d0 values at that step
        """
        if trunc is None:
            trunc = self.config.get('trunc')
        if steps_ahead is None and 'steps_ahead' in self.config:
            steps_ahead = self.config.get('steps_ahead')
            
        if steps_ahead is None:
            return self._format_as_integer_legacy(X, trunc)
        
        results_by_step = {step: [] for step in steps_ahead}
        
        for window in X:
            step_samples = {step: [] for step in steps_ahead}
            for sample in window:
                d0_values = self._extract_d0_values(sample)
                for step in steps_ahead:
                    idx = step - 1
                    if idx < len(d0_values):
                        step_samples[step].append(d0_values[idx])
                    else:
                        step_samples[step].append(None)
            for step in steps_ahead:
                results_by_step[step].append(step_samples[step])
        
        for step in steps_ahead:
            results_by_step[step] = np.array(results_by_step[step], dtype=object)
        
        return results_by_step

    def _extract_d0_values(self, sample):
        """
        Extract all d0 values from a sample string in order.
        For "d0:1,d1:2,d0:3,d1:4", returns [1, 3].
        """
        tokens = re.findall(r'd(\d+):(\d+)', sample)
        d0_values = []
        for dim_str, val_str in tokens:
            if dim_str == "0":
                d0_values.append(int(val_str))
        return d0_values

    def _format_as_integer_legacy(self, X, trunc=None):
        """
        Legacy format_as_integer behavior.
        
        - If trunc is None: returns all values (full round-trip for validation)
        - If trunc is set: extracts only d0 values and truncates (for pipeline)
        """
        batch_rows = []
        for window in X:
            samples = []
            for sample in window:
                if trunc is None:
                    tokens = re.findall(r'd\d+:(\d+)', sample)
                    values = [int(v) for v in tokens]
                else:
                    values = self._extract_d0_values(sample)[:trunc]
                samples.append(values)
            batch_rows.append(samples)
        return np.array(batch_rows, dtype=object)
