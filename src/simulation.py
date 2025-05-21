import numpy as np
import scipy.signal as sig
import pyroomacoustics as pra

SPEED_OF_SOUND = 343  # 声速 (m/s)
SAMPLING_RATE = 16000 # 采样率 (Hz)

class SoundSource:
    """
    表示一个声源及其信号特性。
    """
    def __init__(self, position, name="Source", signal_type="白噪声", signal_params=None):
        """
        初始化声源。
        :param position: 声源位置 [x,y,z]
        :param name: 声源名称
        :param signal_type: 信号类型 (例如 "白噪声", "正弦波组合", "脉冲")
        :param signal_params: 信号参数字典 (具体内容取决于signal_type)
                                e.g., for "正弦波组合": {"components": [{"freq": 440, "amp": 0.5}, ...]}
                                e.g., for "脉冲": {"width": 0.001} (in seconds)
        """
        self.position = np.array(position)
        self.name = name
        self.signal_type = signal_type
        self.signal_params = signal_params if signal_params is not None else {}

    def get_signal(self, duration, sampling_rate=SAMPLING_RATE):
        """
        根据指定的时长和采样率生成声源信号。
        :param duration: 信号时长 (秒)
        :param sampling_rate: 信号采样率 (Hz)
        :return: NumPy array representing the signal.
        """
        num_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        signal = np.zeros(num_samples)

        if self.signal_type == "正弦波组合":
            components = self.signal_params.get("components", [])
            if not components: # Default if no components defined
                components.append({"freq": 440, "amp": 0.7})
            for comp in components:
                freq = comp.get("freq", 440) # Default freq if not specified
                amp = comp.get("amp", 0.5)   # Default amp if not specified
                signal += amp * np.sin(2 * np.pi * freq * t)
            # Normalize if sum of amplitudes is > 1 to prevent clipping, simple normalization
            if components and np.sum([c.get("amp",0) for c in components]) > 1.0:
                max_abs_val = np.max(np.abs(signal))
                if max_abs_val > 0:
                    signal = signal / max_abs_val * 0.98 # Normalize to just below 1.0
        
        elif self.signal_type == "白噪声":
            # Generate Gaussian white noise, scale to approx -1 to 1
            noise_amp = self.signal_params.get("amplitude", 0.5) # Allow configurable amplitude
            signal = noise_amp * np.random.normal(0, 1, num_samples) 
            # Simple clipping to ensure it's within typical audio range, though normalization might be better
            signal = np.clip(signal, -1.0, 1.0)

        elif self.signal_type == "脉冲":
            pulse_width_s = self.signal_params.get("width", 0.001) # Pulse width in seconds
            pulse_width_samples = int(pulse_width_s * sampling_rate)
            if pulse_width_samples > num_samples:
                pulse_width_samples = num_samples # Cap at total duration
            if pulse_width_samples > 0:
                signal[:pulse_width_samples] = 1.0 # Simple rectangular pulse at the beginning
        
        # elif self.signal_type == "from_file": # Placeholder for future
        #     file_path = self.signal_params.get("path", None)
        #     if file_path:
        #         # TODO: Implement loading wav file, resampling if necessary
        #         # from scipy.io import wavfile
        #         # sr_file, sig_file = wavfile.read(file_path)
        #         # if sr_file != sampling_rate: resample...
        #         # signal = loaded_and_resampled_signal[:num_samples]
        #         pass
        else:
            print(f"警告: 未知的信号类型 '{self.signal_type}' for source '{self.name}'. 生成静音。")
            # Fallback to sine if type is unknown or not implemented, or just silence
            # signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        return signal

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
            order = self.cutoff_freqs.get('order', 4)

            if self.freq_response_type == '低通': # Updated to match Chinese type from GUI
                cutoff = self.cutoff_freqs.get('cutoff')
                if cutoff is None or cutoff <= 0 or cutoff >= nyquist:
                    return signal
                b, a = sig.butter(order, cutoff / nyquist, btype='low')
                return sig.lfilter(b, a, signal)
            elif self.freq_response_type == '高通': # Updated to match Chinese type from GUI
                cutoff = self.cutoff_freqs.get('cutoff')
                if cutoff is None or cutoff <= 0 or cutoff >= nyquist:
                    return signal
                b, a = sig.butter(order, cutoff / nyquist, btype='high')
                return sig.lfilter(b, a, signal)
            elif self.freq_response_type == '带通': # Updated to match Chinese type from GUI
                low_cutoff = self.cutoff_freqs.get('low_cutoff')
                high_cutoff = self.cutoff_freqs.get('high_cutoff')

                if low_cutoff is None or high_cutoff is None:
                    return signal
                
                low = low_cutoff / nyquist
                high = high_cutoff / nyquist
                
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

def simulate_with_pyroomacoustics(room_dim, source_objects: list[SoundSource], mic_objects: list[Microphone], duration, rt60=None, material_absorption=0.5):
    # source_signal_data = source_obj.get_signal(duration) # Old: single source

    if rt60 is not None:
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(room_dim, fs=SAMPLING_RATE, materials=pra.Material(e_absorption), max_order=max_order)
    else:
        m = pra.Material(energy_absorption=material_absorption, scattering_coefficient=0.1)
        # Determine max_order based on pyroomacoustics typical behavior for ShoeBox
        # max_order=3 for 2D, or a higher value like 17 for 3D if not specified by rt60
        default_max_order = 3 if len(room_dim) == 2 else 17 
        room = pra.ShoeBox(room_dim, fs=SAMPLING_RATE, materials=m, max_order=default_max_order)

    # Add all sources to the room
    if not source_objects:
        raise ValueError("At least one sound source must be provided.")
        
    for src_obj in source_objects:
        source_signal_data = src_obj.get_signal(duration, sampling_rate=SAMPLING_RATE)
        source_pos_pra = src_obj.position[:len(room_dim)] # Use appropriate dimensions
        room.add_source(source_pos_pra, signal=source_signal_data) # 移除 name 参数，兼容pyroomacoustics

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