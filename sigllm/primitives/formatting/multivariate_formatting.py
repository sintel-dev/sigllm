import numpy as np
import pandas as pd

from sigllm.primitives.formatting.utils import test_multivariate_formatting_validity

class MultivariateFormattingMethod:
    def __init__(self, method_name: str, verbose: bool = False, **kwargs):
        self.method_name = method_name
        self.config = kwargs
        self.metadata = {}
        self.verbose = verbose
        
        if self.method_name != "persistence_control":
            test_multivariate_formatting_validity(self, verbose=verbose)


    def format_as_string(self, data: np.ndarray, **kwargs) -> str:
        raise NotImplementedError()


    def format_as_integer(self, data: str, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df[["timestamp"]]
        vals = df.drop(columns=["timestamp"])
        normed = (vals - vals.mean(axis=0)) / vals.std(axis=0)
        return pd.concat([ts, normed], axis=1)[df.columns] 
