import numpy as np

def evaluate_array_output_conceptual(recorded_signals, ground_truth_signal):
    if not recorded_signals: return np.inf
    num_mics = len(recorded_signals)
    if num_mics == 0: return np.inf

    min_len = min(len(s) for s in recorded_signals.values())
    if min_len == 0: return np.inf

    combined_signal = np.zeros(min_len)
    for signal in recorded_signals.values():
        combined_signal += signal[:min_len]
    combined_signal /= num_mics
    
    if len(ground_truth_signal) > min_len:
        ground_truth_adjusted = ground_truth_signal[:len(combined_signal)]
    elif len(ground_truth_signal) < min_len:
        ground_truth_adjusted = np.pad(ground_truth_signal, (0, min_len - len(ground_truth_signal)))
    else:
        ground_truth_adjusted = ground_truth_signal
        
    mse = np.mean((combined_signal - ground_truth_adjusted)**2)
    return mse 