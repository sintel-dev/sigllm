# -*- coding: utf-8 -*-

"""
Result post-processing module.

This module contains functions that help convert model responses back to timestamps.
"""
import numpy as np
from collections import Counter

def str2ind(text, len_seq, sep=','):
    """Convert a text string to indices.

    Convert a string containing digits into an array of indices.

    Args:
        text (str):
            A string containing indices values.
        len_seq (int): 
            The length of processed sequence 
        sep (str):
            String that was used to separate each element in text, Default to `","`.

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing parsed elements in `text`.
    """
    #Remove all characters from text except the digits and sep
    text = ''.join(i for i in text if (i.isdigit() or i == sep))
    
    values = np.fromstring(text, dtype=int, sep=sep)
    
    #Remove indices that exceed the length of sequence
    values = values[values < len_seq]
    return values


def get_anomaly_list_within_seq(res_list, alpha = 0.5): 
    """Get the final list of anomalous indices of a sequence
    
    Choose which index is considered anomalous in the sequence based on number of votes from multiple LLM responses
    
    Args:
        res_list (list of numpy.ndarray): 
            A list of 1-dimensional array containing anomous indices output by LLM 
        alpha (float): 
            Percentage of total number of votes that an index needs to have to be considered anomalous. Default: 0.5
    Returns:
        numpy.ndarray:
            A 1-dimensional array containing final anomalous indices
    """
    min_vote = np.ceil(alpha*len(res_list))
    
    flattened_res = np.concatenate(res_list)
    
    unique_elements, counts = np.unique(flattened_res, return_counts=True)
    
    final_list = unique_elements[counts >= min_vote]
    
    return final_list

def get_anomaly_list_across_seq(ano_list, window_size, step_size, beta = 0.5):
    """Get the final list of anomalous indices of a sequence when combining all rolling windows
    
    Args: 
        ano_list (list of numpy.ndarray): 
            A list of 1-dimensional array containing anomous indices of each window
        window_size (int): 
            Length of each window 
        step_size (int): 
            Indicating the number of steps the window moves forward each round.
        beta (float): 
            Percentage of number of containing windows that an index needs to have to be considered anomalous. Default: 0.5
    Return: 
        numpy.ndarray:
            A 1-dimensional array containing final anomalous indices        
    """
    min_vote = np.ceil(beta * window_size/step_size)
    
    flattened_res = np.concatenate(ano_list)
    
    unique_elements, counts = np.unique(flattened_res, return_counts=True)
    
    final_list = unique_elements[counts >= min_vote]
    
    return np.sort(final_list)

def ind2time(sequence, ind_list): 
    """Convert list of indices into list of timestamp
    
    Args: 
        sequence (pandas.Dataframe): 
            Signal with timestamps and values
        ind_list (numpy.ndarray): 
            A 1-dimensional array of indices
    Returns: 
        numpy.ndarray:
            A 1-dimensional array containing timestamps of `sequence` corresponding to indices in `ind_list` 
    """
    return sequence.iloc[ind_list].timestamp.to_numpy()


