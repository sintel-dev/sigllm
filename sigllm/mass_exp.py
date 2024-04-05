import pandas as pd
from data import rolling_window_sequences, sig2str
from orion.data import load_signal, load_anomalies
from sigllm import get_anomalies
from gpt import get_gpt_model_response
from prompt import create_message_zero_shot
from anomalies import merge_anomaly_seq
import numpy as np
from urllib.error import HTTPError
from orion.evaluation.utils import from_list_points_timestamps
from orion.evaluation.point import point_confusion_matrix, point_accuracy, point_f1_score, point_precision, point_recall
from orion.evaluation.contextual import contextual_accuracy, contextual_f1_score, contextual_precision, contextual_recall
import pickle

df = pd.read_csv("data_summary.csv") #signal summary df

#signal and result directory
try: 
    df_res = pd.read_csv("gpt-3.5-turbo-2-digits-res.csv")
except: 
    df_res = pd.DataFrame(columns=['signal', 'pkl_file_name'])
    
computed = df_res['signal']

for i, row in df.iterrows(): 
    if row.signal in computed: 
        continue
    true_ano = load_anomalies(row.signal)
    try:
        signal = load_signal(row.signal)
    except HTTPError:
        S3_URL = 'https://sintel-orion-benchmark.s3.amazonaws.com/{}'
        signal = pd.read_csv(S3_URL.format(row.signal + '.csv'))
    values = signal['value'].values
    indices = signal.index.values
    #make rolling windows
    window_size = 2500
    step_size = 500
    windows, start_indices = rolling_window_sequences(values, indices, window_size, step_size)
    #rolling window anomaly detection
    final_ano = []
    i = 0
    error = dict() #to save error (if any) when running each window
    for seq in windows: 
        try:
            final_ano.append(get_anomalies(seq, create_message_zero_shot, get_gpt_model_response, space = True, decimal = 2))
        except Exception as e:
            error[i] = e
        i+= 1
    ano_idx = merge_anomaly_seq(final_ano, start_indices, window_size, step_size, beta = 0)
    anomalies_pts = idx2time(signal, final_res)
    anomalies_contextual = from_list_points_timestamps(anomalies, gap = row.interval) 
     
    ground_truth_pts = []
    ground_truth_context = []
    for i,interval in true_ano.iterrows(): 
        ground_truth_pts += range(interval.start, interval.end +1)
        ground_truth_context.append((interval.start, interval.end))
    
    start, end = (int(signal.iloc[0].timestamp), int(signal.iloc[-1].timestamp))
    #benchmark
    tn, fp, fn, tp = point_confusion_matrix(ground_truth_pts, anomalies_pts, start = start, end = end)
    point_precision = point_precision(ground_truth_pts, anomalies_pts, start = start, end = end)
    point_recall = point_recall(ground_truth_pts, anomalies_pts, start = start, end = end)
    point_accuracy = point_accuracy(ground_truth_pts, anomalies_pts, start = start, end = end)
    point_f1_score = point_f1_score(ground_truth_pts, anomalies_pts, start = start, end = end)
    
    contextual_accuracy = contextual_accuracy(ground_truth_context, anomalies_contextual, start = start, end = end)
    contextual_f1_score = contextual_f1_score(ground_truth_context, anomalies_contextual, start = start, end = end)
    contextual_precision = contextual_precision(ground_truth_context, anomalies_contextual, start = start, end = end)
    contextual_recall = contextual_recall(ground_truth_context, anomalies_contextual, start = start, end = end)
            
    result = {'signal': row.signal, 'error': error, 'anomalies': anomalies_pts, 'tp': tp, 'fp': fp, 
                'fn': fn, 'point_precision': point_precision, 'point_recall': point_recall, 
               'point_accuracy': point_accuracy, 'point_f1': point_f1_score, 
               'contextual_precision': contextual_precision, 'contextual_recall': contextual_recall, 
                'contextual_accuracy': contextual_accuracy, 'contextual_f1': contextual_f1_score}
    file_name = row.signal + 'gpt-3.5-turbo.pickle'
    
    with open(file_name, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    df_res = df_res.append({'signal': row.signal, 'pkl_file_name': file_name})
    
df_res.to_csv("gpt-3.5-turbo-2-digits-res.csv")