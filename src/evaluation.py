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

# --- 新增评估指标 ---
def calculate_c50(rir, sampling_rate):
    # C50 = 10*log10(early/late), early: 0-50ms, late: 50ms-end
    if rir is None or len(rir) == 0:
        return np.nan
    t_50 = int(0.05 * sampling_rate)
    early = np.sum(rir[:t_50]**2)
    late = np.sum(rir[t_50:]**2)
    if late == 0:
        return np.nan
    return 10 * np.log10(early / late)

def calculate_d50(rir, sampling_rate):
    # D50 = early/(early+late), early: 0-50ms, late: 50ms-end
    if rir is None or len(rir) == 0:
        return np.nan
    t_50 = int(0.05 * sampling_rate)
    early = np.sum(rir[:t_50]**2)
    total = np.sum(rir**2)
    if total == 0:
        return np.nan
    return early / total

def calculate_snr(signal, noise):
    # SNR = 10*log10(signal_power/noise_power)
    if signal is None or noise is None or len(signal) == 0 or len(noise) == 0:
        return np.nan
    signal_power = np.mean(np.array(signal)**2)
    noise_power = np.mean(np.array(noise)**2)
    if noise_power == 0:
        return np.nan
    return 10 * np.log10(signal_power / noise_power) 