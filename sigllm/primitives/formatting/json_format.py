import re

import numpy as np

from sigllm.primitives.formatting.multivariate_formatting import MultivariateFormattingMethod


class JSONFormat(MultivariateFormattingMethod):
    """Formatting method that uses JSON-like format with dimension prefixes."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__('json_format', verbose=verbose, **kwargs)

    def format_as_string(self, X: np.ndarray, separator=',', **kwargs) -> str:
        """Format array as string with dimension prefixes."""

        def window_to_json(X):
            rows = []
            for row in X:
                parts = [f'd{i}:{val}' for i, val in enumerate(row)]
                rows.append(','.join(parts))
            return ','.join(rows)

        out = [window_to_json(window) for window in X]
        return out

    def format_as_integer(self, X, trunc=None, steps_ahead=None, target_column=None, **kwargs):
        """Parse model output and extract values for the target column for specified steps ahead.

        Args:
            X (str):
                Model output containing tokens like "d0:1,d1:2,d0:3,d1:4..."
            trunc (int, optional):
                Legacy parameter for truncation (used when steps_ahead is None)
            steps_ahead (list):
                List of step indices to extract (e.g., [1,3,5,10])
                If None, trunc is used to determine the number of values to extract.
            target_column (int):
                Which dimension to extract (default 0). Can also be set via config.

        Returns:
            If steps_ahead is None:
                np.array of shape (batch, samples) with truncated flat values
            If steps_ahead is provided:
                dict mapping step -> np.array of target_column values at that step
        """
        if trunc is None:
            trunc = self.config.get('trunc')
        if steps_ahead is None and 'steps_ahead' in self.config:
            steps_ahead = self.config.get('steps_ahead')
        if target_column is None:
            target_column = self.config.get('target_column', 0)

        if steps_ahead is None:
            return self._format_as_integer_legacy(X, trunc, target_column)

        results_by_step = {step: [] for step in steps_ahead}

        for window in X:
            step_samples = {step: [] for step in steps_ahead}
            for sample in window:
                dim_values = self._extract_dim_values(sample, target_column)
                for step in steps_ahead:
                    idx = step - 1
                    if idx < len(dim_values):
                        step_samples[step].append(dim_values[idx])
                    else:
                        step_samples[step].append(None)
            for step in steps_ahead:
                results_by_step[step].append(step_samples[step])

        for step in steps_ahead:
            results_by_step[step] = np.array(results_by_step[step], dtype=object)

        return results_by_step

    def _format_as_integer_legacy(self, X, trunc=None, target_column=0):
        """Extract values for the target dimension from parsed output.

        Args:
            X (str):
                Model output containing tokens like "d0:1,d1:2,d0:3,d1:4..."
            trunc (int, optional):
                If None, return all values in a 2D array (num_windows, num_samples) where
                    each cell is a list of values for that sample.
                If int, return 3D array (num_windows, num_samples, trunc) taking the first
                    trunc values for each sample. None-padded if trunc is larger
                    than the number of values.
            target_column (int):
                Which dimension to extract (default 0).

        Returns:
            np.array of shape (num_windows, num_samples, num_values)
            or (num_windows, num_samples, trunc) that hold values
            for the target column for each sample in each window.
        """
        if trunc is None:
            batch_rows = []
            for window in X:
                samples = []
                for sample in window:
                    samples.append(self._extract_dim_values(sample, target_column))
                batch_rows.append(samples)
            return np.array(batch_rows, dtype=object)

        num_windows = len(X)
        num_samples = len(X[0]) if num_windows > 0 else 0
        result = np.full((num_windows, num_samples, trunc), fill_value=None)

        for i, window in enumerate(X):
            for j, sample in enumerate(window):
                dim_values = self._extract_dim_values(sample, target_column)
                for k in range(min(trunc, len(dim_values))):
                    result[i, j, k] = dim_values[k]

        return result

    def _extract_dim_values(self, sample, dim):
        """Helper function to extract values for a given column from a sample string in order.

        For "d0:1,d1:2,d0:3,d1:4" with dim=0, returns [1, 3].
        For "d0:1,d1:2,d0:3,d1:4" with dim=1, returns [2, 4].
        """
        tokens = re.findall(r'd(\d+):(\d+)', sample)
        dim_values = []
        for dim_str, val_str in tokens:
            if dim_str == str(dim):
                dim_values.append(int(val_str))
        return dim_values
