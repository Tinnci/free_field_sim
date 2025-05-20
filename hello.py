import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei']


# --- 常量 ---
SPEED_OF_SOUND = 343  # 声速 (m/s) # pyroomacoustics uses its own, but good to have for reference
SAMPLING_RATE = 16000 # 采样率 (Hz) - pyroomacoustics often works well with 16kHz or 8kHz for faster simulation

# --- 迁移说明 ---
# 1. 所有自定义类（SoundSource, Microphone）和函数（simulate_with_pyroomacoustics, evaluate_array_output_conceptual, plot_signals_and_room）
#    将迁移到 src/ 目录下的 modules 中。
# 2. hello.py 只保留 main() 及其调用逻辑，并从 src 导入。
# 3. 入口保持不变。

from src.simulation import SoundSource, Microphone, simulate_with_pyroomacoustics
from src.evaluation import evaluate_array_output_conceptual
from src.visualization import plot_signals_and_room

# --- 主函数：运行仿真示例 ---
def main():
    # 1. 定义声源信号
    def source_sine_wave(t):
        return 0.7 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    
    duration = 0.2  # 秒

    # 2. 定义声源对象
    source1 = SoundSource(position=[2, 3], signal_func=source_sine_wave, name="Source1") # 2D position

    # 3. 定义麦克风对象 (异构)
    mic1 = Microphone(position=[4.0, 4.5], name="Mic1_Broadband", sensitivity=1.0, self_noise_std=0.001)
    mic2 = Microphone(position=[3.5, 1.5], name="Mic2_LowPass_Quiet", sensitivity=0.9, self_noise_std=0.0005,
                      freq_response_type='lowpass', cutoff_freqs=800)
    mic3 = Microphone(position=[4.5, 2.5], name="Mic3_HighSens_HighPass", sensitivity=1.5, self_noise_std=0.002,
                      freq_response_type='highpass', cutoff_freqs=1000)
    
    all_mics = [mic1, mic2, mic3]

    # 4. 定义房间参数 (2D 房间)
    room_dimensions = [6, 5] # [length, width] in meters
    rt60_target = 0.3 # 秒, 目标混响时间

    # 5. 使用 pyroomacoustics 进行仿真并应用麦克风特性
    print(f"使用 pyroomacoustics 进行仿真...")
    recorded_signals_pra, room_obj = simulate_with_pyroomacoustics(
        room_dimensions, source1, all_mics, duration, rt60=rt60_target
    )
    print("仿真完成。")

    # 6. 定义 Ground Truth (简单起见，仍为原始信号)
    ground_truth_signal = source1.get_signal(duration) 

    # 7. 概念性评估
    if recorded_signals_pra:
        mse_pra = evaluate_array_output_conceptual(recorded_signals_pra, ground_truth_signal)
        print(f"pyroomacoustics 阵列的概念性均方误差 (简单平均): {mse_pra:.6f}")
    else:
        print("未能从 pyroomacoustics 获取录制信号。")

    # 8. 可视化
    if recorded_signals_pra:
        plot_signals_and_room(room_obj, ground_truth_signal, recorded_signals_pra, duration, source1)

if __name__ == "__main__":
    main()