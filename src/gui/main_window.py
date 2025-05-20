import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QGridLayout, QGroupBox, QMessageBox,
    QTabWidget, QScrollArea
)
from PySide6.QtCore import Qt

# Matplotlib imports for embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei'] # Ensure font is set for matplotlib plots in GUI

# Project module imports (adjust paths if necessary, assuming src is in PYTHONPATH or using relative imports)
# For simplicity in this direct edit, we might face import issues if not run as a module.
# Proper way is to run `python -m src.gui.main_app` if main_app.py is the entry point.

# Assuming these files are in ../ relative to gui/ during execution as a module, or src. is in path.
try:
    from ..simulation import SoundSource, Microphone, simulate_with_pyroomacoustics, SAMPLING_RATE
    from ..visualization3d import plot_room_3d
    from ..visualization import (
        plot_rir_embed, 
        plot_signals_time_domain_embed, 
        plot_signals_frequency_domain_embed
    )
    # from ..evaluation import evaluate_array_output_conceptual # For evaluation later
except ImportError:
    # Fallback for direct script execution (e.g. python main_window.py from src/gui)
    # This is not ideal for production but helps in development sometimes.
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add src to path
    from simulation import SoundSource, Microphone, simulate_with_pyroomacoustics, SAMPLING_RATE
    from visualization3d import plot_room_3d
    from visualization import (
        plot_rir_embed, 
        plot_signals_time_domain_embed, 
        plot_signals_frequency_domain_embed
    )
    # from evaluation import evaluate_array_output_conceptual


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式声学仿真工具 v0.3")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget) # Main layout changed to QHBoxLayout

        # --- Left Panel: Controls and Parameters ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setFixedWidth(380)

        # Parameters Group
        params_group = QGroupBox("仿真参数设置")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("房间尺寸 (LxWxH, m):"), 0, 0)
        self.room_dims_input = QLineEdit("6,5,3")
        params_layout.addWidget(self.room_dims_input, 0, 1)
        params_layout.addWidget(QLabel("RT60 (s):"), 1, 0)
        self.rt60_input = QLineEdit("0.3")
        params_layout.addWidget(self.rt60_input, 1, 1)
        params_layout.addWidget(QLabel("声源位置 (x,y,z):"), 2, 0)
        self.source_pos_input = QLineEdit("2,3,1.5")
        params_layout.addWidget(self.source_pos_input, 2, 1)
        params_layout.addWidget(QLabel("麦克风位置 (x,y,z;...):"), 3, 0)
        self.mic_pos_input = QLineEdit("4,2.5,1.5; 5,3.5,1.5")
        params_layout.addWidget(self.mic_pos_input, 3, 1)
        params_layout.addWidget(QLabel("信号时长 (s):"), 4, 0)
        self.duration_input = QLineEdit("0.2") # Shorter default for faster testing
        params_layout.addWidget(self.duration_input, 4, 1)
        params_group.setLayout(params_layout)
        left_panel_layout.addWidget(params_group)

        # Control Group
        control_group = QGroupBox("控制")
        control_layout_v = QVBoxLayout()
        self.run_button = QPushButton("运行仿真与更新绘图")
        self.run_button.clicked.connect(self.run_simulation_and_update_plots)
        control_layout_v.addWidget(self.run_button)
        control_group.setLayout(control_layout_v)
        left_panel_layout.addWidget(control_group)
        left_panel_layout.addStretch()
        main_layout.addWidget(left_panel_widget)

        # --- Right Panel: Results Display (Tabbed) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        
        self.tabs = QTabWidget()
        
        # 3D Plot Tab
        self.tab_3d = QWidget()
        self.tab_3d_layout = QVBoxLayout(self.tab_3d)
        self.figure3d = Figure(figsize=(7, 6))
        self.canvas3d = FigureCanvas(self.figure3d)
        self.ax3d = self.figure3d.add_subplot(111, projection='3d')
        self.tab_3d_layout.addWidget(self.canvas3d)
        self.tabs.addTab(self.tab_3d, "3D场景")

        # RIR Plot Tab
        self.tab_rir = QWidget()
        self.tab_rir_layout = QVBoxLayout(self.tab_rir)
        self.figure_rir = Figure(figsize=(7,6))
        self.canvas_rir = FigureCanvas(self.figure_rir)
        self.ax_rir = self.figure_rir.add_subplot(111)
        self.tab_rir_layout.addWidget(self.canvas_rir)
        self.tabs.addTab(self.tab_rir, "房间冲激响应 (RIR)")

        # Time Domain Plot Tab
        self.tab_time = QWidget()
        self.tab_time_layout = QVBoxLayout(self.tab_time)
        self.figure_time = Figure(figsize=(7,6))
        self.canvas_time = FigureCanvas(self.figure_time)
        self.ax_time = self.figure_time.add_subplot(111)
        self.tab_time_layout.addWidget(self.canvas_time)
        self.tabs.addTab(self.tab_time, "时域信号")

        # Frequency Domain Plot Tab
        self.tab_freq = QWidget()
        self.tab_freq_layout = QVBoxLayout(self.tab_freq)
        self.figure_freq = Figure(figsize=(7,6))
        self.canvas_freq = FigureCanvas(self.figure_freq)
        self.ax_freq = self.figure_freq.add_subplot(111)
        self.tab_freq_layout.addWidget(self.canvas_freq)
        self.tabs.addTab(self.tab_freq, "频域信号 (FFT)")
        
        right_panel_layout.addWidget(self.tabs)
        main_layout.addWidget(right_panel_widget)

        self.simulation_room = None # To store the room object from simulation
        self.recorded_signals = None
        self.ground_truth_signal = None
        self.current_duration = 0.2

    def parse_vector_input(self, text_input, dimensions=3):
        parts = text_input.split(',')
        if len(parts) != dimensions:
            raise ValueError(f"需要 {dimensions} 个维度，但得到 {len(parts)} 个: '{text_input}'")
        return [float(p.strip()) for p in parts]

    def parse_multiple_mic_input(self, text_input, dimensions=3):
        mic_strings = text_input.split(';')
        mic_positions = []
        for mic_str in mic_strings:
            mic_str_clean = mic_str.strip()
            if mic_str_clean:
                mic_positions.append(self.parse_vector_input(mic_str_clean, dimensions))
        if not mic_positions:
            raise ValueError("至少需要一个麦克风位置。麦克风位置用分号(;)分隔，每个位置用逗号(,)分隔x,y,z坐标。")
        return mic_positions

    def run_simulation_and_update_plots(self):
        try:
            room_dims_val = self.parse_vector_input(self.room_dims_input.text(), 3)
            rt60_val = float(self.rt60_input.text())
            source_pos_val = self.parse_vector_input(self.source_pos_input.text(), 3)
            mic_positions_val = self.parse_multiple_mic_input(self.mic_pos_input.text(), 3)
            self.current_duration = float(self.duration_input.text())

            if not (len(room_dims_val) == 3):
                 QMessageBox.warning(self, "输入错误", "房间尺寸必须是3D (LxWxH)。")
                 return
            if self.current_duration <= 0:
                QMessageBox.warning(self, "输入错误", "信号时长必须大于0。")
                return

            def source_signal_func(t):
                return 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
            
            source_obj = SoundSource(position=np.array(source_pos_val),
                                     signal_func=source_signal_func,
                                     name="Source1")
            
            mic_objects = []
            for i, pos in enumerate(mic_positions_val):
                mic_objects.append(Microphone(position=np.array(pos), name=f"Mic{i+1}"))

            self.recorded_signals, self.simulation_room = simulate_with_pyroomacoustics(
                room_dim=room_dims_val, 
                source_obj=source_obj, 
                mic_objects=mic_objects, 
                duration=self.current_duration, 
                rt60=rt60_val
            )
            self.ground_truth_signal = source_obj.get_signal(self.current_duration)
            
            # Update 3D Plot
            plot_room_3d(room_dim=room_dims_val, 
                         sources=[source_pos_val], 
                         microphones=mic_positions_val, 
                         ax=self.ax3d)
            self.canvas3d.draw()

            # Update RIR Plot (e.g., for the first microphone and first source)
            if self.simulation_room.rir and self.simulation_room.rir[0] and self.simulation_room.rir[0][0] is not None:
                plot_rir_embed(self.ax_rir, self.simulation_room.rir[0][0], SAMPLING_RATE, 
                               title=f"RIR ({mic_objects[0].name} vs {source_obj.name})")
            else:
                self.ax_rir.clear()
                self.ax_rir.text(0.5, 0.5, '无RIR数据或仿真未包含RIR', horizontalalignment='center', verticalalignment='center')
            self.canvas_rir.draw()

            # Update Time Domain Plot
            plot_signals_time_domain_embed(self.ax_time, self.recorded_signals, 
                                           self.ground_truth_signal, SAMPLING_RATE, self.current_duration,
                                           title="时域信号")
            self.canvas_time.draw()

            # Update Frequency Domain Plot
            plot_signals_frequency_domain_embed(self.ax_freq, self.recorded_signals, 
                                                self.ground_truth_signal, SAMPLING_RATE, 
                                                title="频域信号 (FFT)")
            self.canvas_freq.draw()

            QMessageBox.information(self, "成功", "仿真完成，所有视图已更新！")

            # TODO: Call 2D plotting and update other tabs
            # ground_truth = source_obj.get_signal(duration_val)
            # plot_signals_and_room(self.simulation_room, ground_truth, recorded_signals, duration_val, source_obj)
            # This original function shows multiple plots, need to adapt for GUI embedding one by one.

        except ValueError as e:
            QMessageBox.critical(self, "输入错误", f"参数解析或校验失败: {str(e)}\n请检查所有输入格式和值。")
            print(f"参数错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "仿真或绘图错误", f"发生意外错误: {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback to console for debugging
            print(f"仿真/绘图错误: {e}")

# 用于独立运行 GUI 进行测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 