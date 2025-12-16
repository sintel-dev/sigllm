import numpy as np
from mlblocks import MLPipeline
import pandas as pd
import time
class MultivariateFormattingMethod:
    def __init__(self, method_name: str, verbose: bool = False, **kwargs):
        self.method_name = method_name
        self.config = kwargs
        self.metadata = {}
        self.verbose = verbose
        
        if self.method_name != "persistence_control":
            self.test_multivariate_formatting_validity(verbose=verbose)


    def format_as_string(self, data: np.ndarray, **kwargs) -> str:
        raise NotImplementedError()


    def format_as_integer(self, data: str, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df[["timestamp"]]
        vals = df.drop(columns=["timestamp"])
        normed = (vals - vals.mean(axis=0)) / vals.std(axis=0)
        return pd.concat([ts, normed], axis=1)[df.columns] 


    @staticmethod
    def create_test_data(N = 25):
        x1 = np.linspace(10, 9+N, N) / 100
        x2 = np.array([i % 2 for i in range(N)])
        x3 = np.linspace(N+40, 41, N) / 100

        return pd.DataFrame({
            'timestamp': np.linspace(0, 3600*(N-1), N),
            'x1': x1,
            'x2': x2,
            'x3': x3,
        })


    def run_pipeline(self, data=create_test_data(), 
                            interval=3600, 
                            window_size=15, 
                            verbose=True, 
                            samples=7, 
                            normalize=False, 
                            temp=0.1, 
                            return_y_hat = False, 
                            multivariate_allowed_symbols = [],
                            pipeline_name = 'mistral_detector',
                            stride = 1,
                            n_clusters = 2,
                            strategy = 'scaling',
                            steps_ahead = None):
        """
        Run the forecasting pipeline.
        """
        pipeline = MLPipeline(pipeline_name)
        digits_per_timestamp = self.config.get('digits_per_timestamp', 2)
        
        num_dims = len(data.columns) - 1
        
        if steps_ahead is not None:
            max_steps = max(steps_ahead)
            hf_steps = max_steps * (num_dims + 1) # adding some padding here
        else:
            hf_steps = 2

        test_hyperparameters = {
            "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                "interval": interval
            },
            "sigllm.primitives.forecasting.custom.rolling_window_sequences#1": {
                "target_column": 0,
                "window_size": window_size,
                "target_size": max(steps_ahead) if steps_ahead else 1,
                "step_size": stride,
            },
            "sigllm.primitives.forecasting.huggingface.HF#1": {
                "samples": samples,
                "temp": temp,
                "multivariate_allowed_symbols": multivariate_allowed_symbols,
                "steps": hf_steps,
            },
        }
        
        if strategy == 'binning':
            test_hyperparameters["sigllm.primitives.transformation.Float2Scalar#1"] = {
                "strategy": "binning",
                "n_clusters": n_clusters,
            }

        elif strategy == 'scaling':
            test_hyperparameters["sigllm.primitives.transformation.Float2Scalar#1"] = {
                "decimal": digits_per_timestamp,
                "rescale": True,
            }
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        print("STARTING PIPELINE: ")
        time.sleep(10)

        pipeline.set_hyperparameters(test_hyperparameters)
        if normalize:
            data = self.normalize_data(data)
        context = pipeline.fit(data, start_=0, output_=3)
        context['X'] = self.format_as_string(context['X'], **self.config)

        if self.method_name == "persistence_control":
            context['y_hat'] = context['X']
            
        else:
            context = pipeline.fit(**context, start_=5, output_=5)
        
        if verbose:
            print(f"y_hat example: {context['y_hat'][0][0]}")

        if steps_ahead is not None:
            return self._process_multi_step_results(
                context, pipeline, steps_ahead, return_y_hat, verbose
            )

        context['y_hat'] = self.format_as_integer(context['y_hat'], trunc=1)
        if verbose:
            print(f"y_hat example: {context['y_hat'][0][0]}")
        context = pipeline.fit(**context, start_=7, output_=10)

        errors = np.round(context['errors'], 7)

        if verbose:
            print(f"y_hat: {context['y_hat']}")
            print(f"y: {context['y']}")
            print(f"errors: {errors}")

        if return_y_hat:
            return errors, context['y_hat'], context['y']
        else:
            return errors

    def _process_multi_step_results(self, context, pipeline, steps_ahead, return_y_hat, verbose):
        """
        Process results for multi-step-ahead prediction.
        
        For multi-step predictions with stride > 1, we skip aggregate_rolling_window
        since there's no overlap between predictions. Each window gives one prediction
        per step, indexed sequentially (0, 1, 2, ...) regardless of actual stride.
        
        Returns:
            dict {step : {'errors': [...], 'y_hat': [...], 'y': [...]}}
        """
        y_hat_by_step = self.format_as_integer(
            context['y_hat'], steps_ahead=steps_ahead
        )
        
        results = {}
        
        for step in steps_ahead:
            step_context = context.copy()
            
            y_hat_step = y_hat_by_step[step]
            y_hat_float = np.array([[v if v is not None else np.nan for v in row] for row in y_hat_step], dtype=float)
            step_context['y_hat'] = np.expand_dims(y_hat_float, axis=-1)
            step_context = pipeline.fit(**step_context, start_=7, output_=7)
            # Aggregate across samples using median, then squeeze
            y_hat_agg = np.nanmedian(step_context['y_hat'], axis=1).squeeze() 
            
            # Get ground truth for this step
            y_for_step = context['y'][:, step - 1] if context['y'].ndim > 1 else context['y']
            
            # Compute errors directly (residuals)
            errors = np.round(y_hat_agg - y_for_step, 7)
            
            if verbose:
                print(f"Step {step} - y_hat shape: {y_hat_agg.shape}, errors shape: {errors.shape}")
            
            results[step] = {
                'errors': errors,
                'y_hat': y_hat_agg,
                'y': y_for_step,
            }
        
        if return_y_hat:
            return results
        else:
            return {step: results[step]['errors'] for step in steps_ahead}
    

    def test_multivariate_formatting_validity(self, data=None, verbose=False):
        if verbose:
            print("Testing multivariate formatting method validity")

        if data is None:
            raw_data = np.array(self.create_test_data())[:, 1:]
            windowed_data = np.array([raw_data[i:i+15,:] for i in range(0, len(raw_data)-15, 1)])
            data = (1000 * windowed_data).astype(int)
            if verbose:
                print(data.shape)

        string_data = self.format_as_string(data, **self.config)
        LLM_mock_output = np.array(string_data).reshape(-1, 1)
        if verbose:
            print(LLM_mock_output)
        integer_data = self.format_as_integer(LLM_mock_output, **self.config)
        if verbose:
            print(f"Format as string output: {string_data}")

        assert isinstance(string_data, list)
        assert isinstance(string_data[0], str)
        assert isinstance(integer_data, np.ndarray)

        if self.method_name == "univariate_control":
            assert np.all(integer_data.flatten() == data[:, :, 0].flatten())
        else:
            assert np.all(integer_data.flatten() == data.flatten())

        if verbose:
            print("Validation suite passed")


if __name__ == "__main__":
    method = MultivariateFormattingMethod(method_name="test")
    print(method.normalize_data(method.create_test_data()))
