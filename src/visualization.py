import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei']

# SAMPLING_RATE should ideally be imported from a central config or simulation module
# For now, embeddable functions will require it as an argument.

def plot_rir_embed(ax, rir_data, sampling_rate, title="Room Impulse Response"):
    """Plots RIR on a given Matplotlib Axes object for GUI embedding."""
    ax.clear()
    if rir_data is not None and len(rir_data) > 0:
        time_axis = np.arange(len(rir_data)) / sampling_rate
        ax.plot(time_axis, rir_data)
    ax.set_title(title)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度")
    ax.grid(True)
    ax.figure.tight_layout() # Adjust layout

def plot_signal_time_domain_embed(ax, signal_data, sampling_rate, duration, label, title="Time Domain Signal", color=None):
    """Plots a single time-domain signal on a given Matplotlib Axes object."""
    ax.clear()
    if signal_data is not None and len(signal_data) > 0:
        time_axis = np.linspace(0, duration, len(signal_data), endpoint=False)
        ax.plot(time_axis, signal_data, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度")
    ax.grid(True)
    if label:
        ax.legend()
    ax.figure.tight_layout()

def plot_signals_time_domain_embed(ax, signals_dict, ground_truth_signal, sampling_rate, duration, title="Time Domain Signals"):
    """Plots multiple time-domain signals on a given Matplotlib Axes object."""
    ax.clear()
    
    # Plot Ground Truth
    if ground_truth_signal is not None and len(ground_truth_signal) > 0:
        gt_time_axis = np.linspace(0, duration, len(ground_truth_signal), endpoint=False)
        ax.plot(gt_time_axis, ground_truth_signal, label="Ground Truth (原始声源)", color='black', linestyle='--')

    # Plot Recorded Signals
    if signals_dict:
        max_len = 0
        for sig in signals_dict.values():
            if sig is not None and len(sig) > max_len:
                max_len = len(sig)
        
        if max_len > 0:
            time_axis_recorded = np.linspace(0, duration, max_len, endpoint=False)
            for name, signal in signals_dict.items():
                if signal is not None and len(signal) > 0:
                    # Ensure all signals are plotted against the longest common time axis if lengths differ slightly due to processing
                    current_signal_time_axis = np.linspace(0, duration, len(signal), endpoint=False)
                    ax.plot(current_signal_time_axis, signal, label=f"Mic: {name}")
    
    ax.set_title(title)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度")
    ax.grid(True)
    ax.legend()
    ax.figure.tight_layout()

def plot_signal_frequency_domain_embed(ax, signal_data, sampling_rate, label, title="Frequency Domain Signal", color=None):
    """Plots a single frequency-domain signal (FFT) on a given Matplotlib Axes object."""
    ax.clear()
    if signal_data is not None and len(signal_data) > 0:
        fft_signal = np.fft.fft(signal_data)
        freq_axis = np.fft.fftfreq(len(fft_signal), d=1./sampling_rate)
        half_len = len(freq_axis) // 2
        ax.plot(freq_axis[:half_len], np.abs(fft_signal)[:half_len], label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度谱")
    ax.grid(True)
    if label:
        ax.legend()
    ax.figure.tight_layout()

def plot_signals_frequency_domain_embed(ax, signals_dict, ground_truth_signal, sampling_rate, title="Frequency Domain Signals"):
    """Plots multiple frequency-domain signals on a given Matplotlib Axes object."""
    ax.clear()

    # Plot Ground Truth FFT
    if ground_truth_signal is not None and len(ground_truth_signal) > 0:
        fft_gt = np.fft.fft(ground_truth_signal)
        freq_gt = np.fft.fftfreq(len(fft_gt), d=1./sampling_rate)
        ax.plot(freq_gt[:len(freq_gt)//2], np.abs(fft_gt)[:len(fft_gt)//2], label="Ground Truth (FFT)", color='black', linestyle='--')

    # Plot Recorded Signals FFT
    if signals_dict:
        for name, signal in signals_dict.items():
            if signal is not None and len(signal) > 0:
                fft_signal = np.fft.fft(signal)
                freq_signal = np.fft.fftfreq(len(fft_signal), d=1./sampling_rate)
                ax.plot(freq_signal[:len(freq_signal)//2], np.abs(fft_signal)[:len(freq_signal)//2], label=f"Mic: {name} (FFT)")

    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度谱")
    ax.grid(True)
    ax.legend()
    ax.figure.tight_layout()


# --- Original plot_signals_and_room (kept for non-GUI use or future refactoring) ---
# This function creates its own figures and uses plt.show().
# It is NOT directly used for embedding individual plots into the GUI tabs.

def plot_signals_and_room(room, ground_truth_signal, recorded_signals, duration, source_obj, current_sampling_rate):
    # This original function is complex and produces multiple figures via plt.show().
    # For GUI, we are using the dedicated _embed functions above to plot into specific tabs.
    # If you need to run this function standalone, it will still work as before.
    print("Executing original plot_signals_and_room function (generates separate figures)...")

    # 1. Room layout (pyroomacoustics own plot)
    if hasattr(room, 'plot'):
        try:
            fig_room = room.plot(img_order=0) # In new pyroomacoustics, this might return fig, ax
            if fig_room: # If it returns a figure or (fig,ax) tuple
                plt.figure(fig_room.number) # Ensure it's the current figure
                plt.title("房间布局、声源和麦克风阵列 (原始视图)")
                plt.show()
        except Exception as e:
            print(f"Error plotting room layout with room.plot(): {e}")

    # 2. RIR (first mic, first source)
    if room.rir and room.rir[0] and room.rir[0][0] is not None:
        fig_rir, ax_rir = plt.subplots(figsize=(12,4))
        plot_rir_embed(ax_rir, room.rir[0][0], current_sampling_rate, title=f"RIR - {list(recorded_signals.keys())[0]} vs {source_obj.name}")
        fig_rir.tight_layout()
        plt.show()

    # 3. Time Domain Signals
    if recorded_signals or ground_truth_signal is not None:
        fig_time, ax_time = plt.subplots(figsize=(15, 6))
        plot_signals_time_domain_embed(ax_time, recorded_signals, ground_truth_signal, current_sampling_rate, duration, title="All Time Domain Signals")
        fig_time.tight_layout()
        plt.show()

    # 4. Frequency Domain Signals
    if recorded_signals or ground_truth_signal is not None:
        fig_freq, ax_freq = plt.subplots(figsize=(15, 6))
        plot_signals_frequency_domain_embed(ax_freq, recorded_signals, ground_truth_signal, current_sampling_rate, title="All Frequency Domain Signals (FFT)")
        fig_freq.tight_layout()
        plt.show() 