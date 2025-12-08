from .multivariate_formatting import MultivariateFormattingMethod
import numpy as np


class DigitInterleave(MultivariateFormattingMethod):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__("digit_interleave", verbose=verbose, **kwargs)


    def format_as_string(self, data: np.ndarray, digits_per_timestamp = 3, separator = ",") -> str:     
        max_digits = max(len(str(abs(int(v)))) for window in data for ts in window for v in ts)
        width_used = max(digits_per_timestamp, max_digits)
        self.metadata['width_used'] = width_used

        def interleave_digits(timestamp):
            str_values = [str(int(val)) for val in timestamp]
            padded_values = [s.zfill(width_used) for s in str_values]
            result_str = ''
            for digit_pos in range(width_used):
                for padded_val in padded_values:
                    result_str += padded_val[digit_pos]

            return result_str

        result = [
            separator.join(interleave_digits(timestamp) for timestamp in window) + separator  # Add comma at the end
            for window in data
        ]
        return result


    def format_as_integer(self, data: list[str], separator = ",", trunc = None, digits_per_timestamp = 3) -> np.ndarray:
        width_used = self.metadata['width_used']

        def deinterleave_timestamp(interleaved_str):
            """Convert interleaved digits back to original values"""
            total_digits = len(interleaved_str)
            num_values = total_digits // width_used

            # Reconstruct each original value
            values = []
            for value_idx in range(num_values):
                # Collect digits for this value from each position
                value_digits = []
                for digit_pos in range(width_used):
                    # Calculate position in interleaved string
                    pos = digit_pos * num_values + value_idx
                    if pos < total_digits:
                        value_digits.append(interleaved_str[pos])

                if value_digits:
                    values.append(int(''.join(value_digits)))

            return np.array(values)[:trunc] if trunc else np.array(values)

        result = np.array([
            [
                deinterleave_timestamp(timestamp)
                for sample in entry
                for timestamp in sample.lstrip(separator).rstrip(separator).split(separator)[:trunc]
                if timestamp.strip()  # Skip empty strings
            ]
            for entry in data
        ], dtype=object)
        return result



if __name__ == "__main__":
    method = DigitInterleave(digits_per_timestamp=3)
    method.test_multivariate_formatting_validity(verbose=False)
    errs, y_hat, y = method.run_pipeline(return_y_hat=True)
    print(errs)
    print(y_hat)
    print(y)