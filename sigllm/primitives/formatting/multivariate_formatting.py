import numpy as np
import pandas as pd


class MultivariateFormattingMethod:
    """Base class for multivariate formatting methods.

    Subclasses implement format_as_string and format_as_integer to convert
    between numpy arrays and string representations for LLM input/output.

    The target_column parameter (default 0) can be passed to subclass methods
    or set via config to specify which dimension to extract. This should
    match the target_column parameter used in rolling_window_sequences.
    """

    def __init__(self, method_name: str, verbose: bool = False, **kwargs):
        self.method_name = method_name
        self.config = kwargs
        self.metadata = {}
        self.verbose = verbose

    def format_as_string(self, X: np.ndarray, **kwargs) -> str:
        """Format array as string representation."""
        raise NotImplementedError()

    def format_as_integer(self, X: str, **kwargs) -> np.ndarray:
        """Parse string representation back to integer array."""
        raise NotImplementedError()

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data by subtracting mean and dividing by std."""
        ts = df[['timestamp']]
        vals = df.drop(columns=['timestamp'])
        normed = (vals - vals.mean(axis=0)) / vals.std(axis=0)
        return pd.concat([ts, normed], axis=1)[df.columns]
