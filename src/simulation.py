import numpy as np
import scipy.signal as sig
import pyroomacoustics as pra

SPEED_OF_SOUND = 343  # 声速 (m/s)
SAMPLING_RATE = 16000 # 采样率 (Hz)

class SoundSource:
    """
    表示一个声源。
    """
    def __init__(self, position, signal_func, name="Source"):
        self.position = np.array(position)
        self.signal_func = signal_func
        self.name = name

    def get_signal(self, duration):
        t = np.linspace(0, duration, int(duration * SAMPLING_RATE), endpoint=False)
        return self.signal_func(t)

class Microphone:
    """
    表示一个麦克风及其特性。
    声学传播由 pyroomacoustics 处理，此类用于应用后续特性。
    """
    def __init__(self, position, name="Mic", sensitivity=1.0, self_noise_std=0.01,
                 freq_response_type=None, cutoff_freqs=None):
        self.position = np.array(position)
        self.name = name
        self.sensitivity = sensitivity
        self.self_noise_std = self_noise_std
        self.freq_response_type = freq_response_type
        self.cutoff_freqs = cutoff_freqs

    def _apply_frequency_response(self, signal, current_sampling_rate):
        if self.freq_response_type and self.cutoff_freqs:
            nyquist = 0.5 * current_sampling_rate
            order = 4
            if self.freq_response_type == 'lowpass':
                if self.cutoff_freqs <= 0 or self.cutoff_freqs >= nyquist:
                    return signal
                b, a = sig.butter(order, self.cutoff_freqs / nyquist, btype='low')
                return sig.lfilter(b, a, signal)
            elif self.freq_response_type == 'highpass':
                if self.cutoff_freqs <= 0 or self.cutoff_freqs >= nyquist:
                    return signal
                b, a = sig.butter(order, self.cutoff_freqs / nyquist, btype='high')
                return sig.lfilter(b, a, signal)
            elif self.freq_response_type == 'bandpass':
                low = self.cutoff_freqs[0] / nyquist
                high = self.cutoff_freqs[1] / nyquist
                if low <= 0 or high >= 1 or low >= high:
                    return signal
                b, a = sig.butter(order, [low, high], btype='band')
                return sig.lfilter(b, a, signal)
        return signal

    def apply_mic_characteristics(self, signal_from_room_sim, current_sampling_rate):
        sensitive_signal = signal_from_room_sim * self.sensitivity
        responded_signal = self._apply_frequency_response(sensitive_signal, current_sampling_rate)
        noise = np.random.normal(0, self.self_noise_std, len(responded_signal))
        final_signal = responded_signal + noise
        return final_signal

def simulate_with_pyroomacoustics(room_dim, source_obj, mic_objects, duration, rt60=None, material_absorption=0.5):
    source_signal_data = source_obj.get_signal(duration)

    if rt60 is not None:
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(room_dim, fs=SAMPLING_RATE, materials=pra.Material(e_absorption), max_order=max_order)
    else:
        m = pra.Material(energy_absorption=material_absorption, scattering_coefficient=0.1)
        room = pra.ShoeBox(room_dim, fs=SAMPLING_RATE, materials=m, max_order=3 if len(room_dim)==2 else 17)

    source_pos_pra = source_obj.position[:len(room_dim)]
    room.add_source(source_pos_pra, signal=source_signal_data)

    mic_positions_pra = []
    for mic in mic_objects:
        mic_positions_pra.append(mic.position[:len(room_dim)])
    mic_array_locs = np.array(mic_positions_pra).T
    room.add_microphone_array(mic_array_locs)

    room.simulate()

    processed_signals = {}
    for i, mic_obj in enumerate(mic_objects):
        signal_at_mic_clean = room.mic_array.signals[i, :]
        processed_signals[mic_obj.name] = mic_obj.apply_mic_characteristics(signal_at_mic_clean, SAMPLING_RATE)

    return processed_signals, room 