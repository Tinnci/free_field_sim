import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QGridLayout, QGroupBox, QMessageBox,
    QTabWidget, QListWidget, QListWidgetItem, QInputDialog, QDialog, QDialogButtonBox,
    QSizePolicy # Added for sizing policy
)
from PySide6.QtCore import Qt

# Matplotlib imports for embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei'] # Ensure font is set for matplotlib plots in GUI

# PyVista imports
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor # Use QtInteractor for embedding
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    QtInteractor = None
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista or pyvistaqt not found. 3D visualization will be disabled or use Matplotlib fallback if implemented.")

# Project module imports (adjust paths if necessary, assuming src is in PYTHONPATH or using relative imports)
# For simplicity in this direct edit, we might face import issues if not run as a module.
# Proper way is to run `python -m src.gui.main_app` if main_app.py is the entry point.

# Assuming these files are in ../ relative to gui/ during execution as a module, or src. is in path.
try:
    from ..simulation import SoundSource, Microphone, simulate_with_pyroomacoustics, SAMPLING_RATE
    from ..visualization3d import create_pyvista_scene, PYVISTA_AVAILABLE as VIZ3D_PYVISTA_AVAILABLE # Import new PyVista function
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
    from visualization3d import create_pyvista_scene, PYVISTA_AVAILABLE as VIZ3D_PYVISTA_AVAILABLE
    from visualization import (
        plot_rir_embed, 
        plot_signals_time_domain_embed, 
        plot_signals_frequency_domain_embed
    )
    # from evaluation import evaluate_array_output_conceptual

# Ensure a consistent check for PyVista availability
if not PYVISTA_AVAILABLE and VIZ3D_PYVISTA_AVAILABLE:
    # This case implies pyvista was found by visualization3d.py but not here, or vice versa.
    # For simplicity, rely on the one from visualization3d as it's closer to the pv-specific code.
    PYVISTA_AVAILABLE = VIZ3D_PYVISTA_AVAILABLE

class PositionInputDialog(QDialog):
    def __init__(self, parent=None, title="输入位置", prompt="请输入位置 (x,y,z):"):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.prompt_label = QLabel(prompt)
        layout.addWidget(self.prompt_label)
        self.position_input = QLineEdit()
        layout.addWidget(self.position_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_position_text(self):
        return self.position_input.text()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式声学仿真工具 v0.4")
        self.setGeometry(100, 100, 1300, 850)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left Panel: Controls and Parameters ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setFixedWidth(450)

        # Parameters Group
        params_group = QGroupBox("仿真参数设置")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("房间尺寸 (LxWxH, m):"), 0, 0, 1, 2)
        self.room_dims_input = QLineEdit("6,5,3")
        params_layout.addWidget(self.room_dims_input, 1, 0, 1, 2)
        params_layout.addWidget(QLabel("RT60 (s):"), 2, 0, 1, 2)
        self.rt60_input = QLineEdit("0.3")
        params_layout.addWidget(self.rt60_input, 3, 0, 1, 2)
        params_layout.addWidget(QLabel("信号时长 (s):"), 4, 0, 1, 2)
        self.duration_input = QLineEdit("0.2")
        params_layout.addWidget(self.duration_input, 5, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        left_panel_layout.addWidget(params_group)

        # Sources Group
        sources_group = QGroupBox("声源管理")
        sources_layout = QVBoxLayout()
        self.sources_list_widget = QListWidget()
        self.sources_list_widget.setFixedHeight(100)
        sources_layout.addWidget(self.sources_list_widget)
        sources_buttons_layout = QHBoxLayout()
        self.add_source_button = QPushButton("添加声源")
        self.add_source_button.clicked.connect(self.add_source)
        self.remove_source_button = QPushButton("移除选中声源")
        self.remove_source_button.clicked.connect(self.remove_source)
        sources_buttons_layout.addWidget(self.add_source_button)
        sources_buttons_layout.addWidget(self.remove_source_button)
        sources_layout.addLayout(sources_buttons_layout)
        sources_group.setLayout(sources_layout)
        left_panel_layout.addWidget(sources_group)
        # Add default source
        self.sources_list_widget.addItem("2,3,1.5")

        # Microphones Group
        mics_group = QGroupBox("麦克风管理")
        mics_layout = QVBoxLayout()
        self.mics_list_widget = QListWidget()
        self.mics_list_widget.setFixedHeight(100)
        mics_layout.addWidget(self.mics_list_widget)
        mics_buttons_layout = QHBoxLayout()
        self.add_mic_button = QPushButton("添加麦克风")
        self.add_mic_button.clicked.connect(self.add_mic)
        self.remove_mic_button = QPushButton("移除选中麦克风")
        self.remove_mic_button.clicked.connect(self.remove_mic)
        mics_buttons_layout.addWidget(self.add_mic_button)
        mics_buttons_layout.addWidget(self.remove_mic_button)
        mics_layout.addLayout(mics_buttons_layout)
        mics_group.setLayout(mics_layout)
        left_panel_layout.addWidget(mics_group)
        # Add default microphones
        self.mics_list_widget.addItem("4,2.5,1.5")
        self.mics_list_widget.addItem("5,3.5,1.5")

        # Selected Object Info Group
        selected_info_group = QGroupBox("选中对象信息")
        selected_info_layout = QVBoxLayout()
        self.selected_object_info_label = QLabel("点击3D视图中的声源或麦克风以查看信息")
        self.selected_object_info_label.setWordWrap(True)
        self.selected_object_info_label.setAlignment(Qt.AlignTop)
        self.selected_object_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        selected_info_layout.addWidget(self.selected_object_info_label)
        selected_info_group.setLayout(selected_info_layout)
        left_panel_layout.addWidget(selected_info_group)

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
        
        # 3D Plot Tab - Now using PyVista
        self.tab_3d = QWidget()
        self.tab_3d_layout = QVBoxLayout(self.tab_3d)
        if PYVISTA_AVAILABLE and QtInteractor is not None:
            # Using QtInteractor for embedded 3D view
            # QtInteractor is a QWidget, can be added directly.
            # It also acts as a plotter, so create_pyvista_scene can use it.
            self.pv_plotter = QtInteractor(self.tab_3d, auto_update=True)
            # self.pv_plotter.set_background('lightgray') # Set background if desired
            self.tab_3d_layout.addWidget(self.pv_plotter) # Add QtInteractor widget directly
        else:
            # Fallback or placeholder if PyVista is not available
            fallback_label = QLabel("PyVista 3D Plotter (QtInteractor) 不可用。\n请安装 PyVista 和 pyvistaqt.")
            fallback_label.setAlignment(Qt.AlignCenter)
            self.tab_3d_layout.addWidget(fallback_label)
            self.pv_plotter = None
        self.tabs.addTab(self.tab_3d, "3D场景 (PyVista)")

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
        self.current_source_positions = [] # Cache for PyVista picking
        self.current_mic_positions = []    # Cache for PyVista picking

    def parse_vector_input(self, text_input, dimensions=3):
        parts = text_input.split(',')
        if len(parts) != dimensions:
            raise ValueError(f"需要 {dimensions} 个维度，但得到 {len(parts)} 个: '{text_input}'")
        return [float(p.strip()) for p in parts]

    def parse_item_list_positions(self, list_widget, dimensions=3):
        positions = []
        for i in range(list_widget.count()):
            item_text = list_widget.item(i).text()
            try:
                positions.append(self.parse_vector_input(item_text, dimensions))
            except ValueError as e:
                raise ValueError(f"列表中的无效位置 '{item_text}': {e}")
        return positions

    def add_item_to_list_widget(self, list_widget, item_type_name):
        dialog = PositionInputDialog(self, title=f"添加{item_type_name}", prompt=f"请输入{item_type_name}位置 (x,y,z):")
        if dialog.exec() == QDialog.Accepted:
            pos_text = dialog.get_position_text()
            try:
                # Validate format before adding
                self.parse_vector_input(pos_text, 3) 
                list_widget.addItem(pos_text)
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"无效的{item_type_name}位置格式: {pos_text}\n{e}\n请输入类似 '1,2,3' 的格式。")

    def add_source(self):
        self.add_item_to_list_widget(self.sources_list_widget, "声源")

    def remove_source(self):
        selected_items = self.sources_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要移除的声源。")
            return
        for item in selected_items:
            self.sources_list_widget.takeItem(self.sources_list_widget.row(item))
        if self.sources_list_widget.count() == 0: # Ensure at least one source if all are removed
            QMessageBox.information(self, "提示", "至少需要一个声源。已自动添加默认声源。")
            self.sources_list_widget.addItem("2,3,1.5")

    def add_mic(self):
        self.add_item_to_list_widget(self.mics_list_widget, "麦克风")

    def remove_mic(self):
        selected_items = self.mics_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要移除的麦克风。")
            return
        for item in selected_items:
            self.mics_list_widget.takeItem(self.mics_list_widget.row(item))
        if self.mics_list_widget.count() == 0: # Ensure at least one mic if all are removed
            QMessageBox.information(self, "提示", "至少需要一个麦克风。已自动添加默认麦克风。")
            self.mics_list_widget.addItem("4,2.5,1.5")

    def run_simulation_and_update_plots(self):
        try:
            room_dims_val = self.parse_vector_input(self.room_dims_input.text(), 3)
            rt60_val = float(self.rt60_input.text())
            
            source_positions_val = self.parse_item_list_positions(self.sources_list_widget)
            mic_positions_val = self.parse_item_list_positions(self.mics_list_widget)
            
            self.current_duration = float(self.duration_input.text())

            if not source_positions_val:
                QMessageBox.warning(self, "输入错误", "至少需要一个声源。")
                return
            if not mic_positions_val:
                QMessageBox.warning(self, "输入错误", "至少需要一个麦克风。")
                return
            
            if not (len(room_dims_val) == 3):
                 QMessageBox.warning(self, "输入错误", "房间尺寸必须是3D (LxWxH)。")
                 return
            if self.current_duration <= 0:
                QMessageBox.warning(self, "输入错误", "信号时长必须大于0。")
                return

            def source_signal_func(t):
                return 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
            
            # For now, we use the first source's position and signal for the simulation core.
            # Pyroomacoustics can support multiple sources, but our SoundSource class and 
            # simulate_with_pyroomacoustics currently assume one primary source object for signal generation.
            # This will need to be refactored if multiple distinct source signals are required simultaneously.
            # The 3D plot will show all source positions.
            
            if not source_positions_val: # Should have been caught earlier, but double check
                 QMessageBox.warning(self, "配置错误", "没有定义声源。")
                 return

            # Use the first source for the simulation's primary signal characteristics
            primary_source_pos_val = source_positions_val[0]
            source_obj = SoundSource(position=np.array(primary_source_pos_val),
                                     signal_func=source_signal_func,
                                     name="Source1") # Name could be dynamic if sources have names
            
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
            
            # Update 3D Plot with PyVista
            if self.pv_plotter is not None and PYVISTA_AVAILABLE:
                create_pyvista_scene(self.pv_plotter, 
                                     room_dim=room_dims_val, 
                                     sources=source_positions_val, 
                                     microphones=mic_positions_val)
                self.current_source_positions = source_positions_val # Cache for picking
                self.current_mic_positions = mic_positions_val     # Cache for picking
                
                # Setup picking callback for PyVista QtInteractor
                # The enable_actor_picking might be slightly different or might work directly
                # on QtInteractor as it inherits from BasePlotter.
                if hasattr(self.pv_plotter, 'enable_actor_picking'):
                    self.pv_plotter.enable_actor_picking(callback=self.on_pick_pyvista, show_message=False)
                else:
                    print("QtInteractor does not have enable_actor_picking directly, check documentation or alternative.")
                
                # self.pv_plotter.app.processEvents() # Not typically needed for QtInteractor as it's part of the Qt app event loop
                self.pv_plotter.update() # Ensure the scene is rendered
            else:
                # Matplotlib fallback logic or warning if neither is available
                # self.ax3d.clear()
                # returned_figure, self.plotted_3d_elements = plot_room_3d(room_dim=room_dims_val, 
                #              sources=source_positions_val, 
                #              microphones=mic_positions_val, 
                #              ax=self.ax3d)
                # self.source_positions_cache = source_positions_val
                # self.mic_positions_cache = mic_positions_val
                # if hasattr(self, '_picker_cid') and self._picker_cid is not None:
                #     self.canvas3d.mpl_disconnect(self._picker_cid)
                # self._picker_cid = self.canvas3d.mpl_connect('pick_event', self.on_pick_3d)
                # self.canvas3d.draw()
                print("PyVista plotter not available for 3D scene.")

            # Update RIR Plot (e.g., for the first microphone and first source)
            # Note: RIR is typically source-mic pair specific.
            # If multiple sources are active in pyroomacoustics, RIR handling becomes more complex.
            # Current simulation uses one source signal, so RIR is from that source.
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

    def on_pick_pyvista(self, picked_actor, *args):
        info_text = "点击了3D场景中的空白区域。"
        if picked_actor is None:
            self.selected_object_info_label.setText(info_text)
            return

        actor_name = picked_actor.name if hasattr(picked_actor, 'name') else "Unknown Actor"
        
        # Determine if it's a source or microphone based on the name given in create_pyvista_scene
        if actor_name.startswith("source_"):
            try:
                idx = int(actor_name.split('_')[1])
                if idx < len(self.current_source_positions):
                    pos = self.current_source_positions[idx]
                    info_text = f"选中的声源:\n类型: 声源\n拾取名称: {actor_name}\n索引: {idx}\n位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m"
                else:
                    info_text = f"选中的声源索引 ({idx}) 超出范围。"
            except (IndexError, ValueError) as e:
                info_text = f"解析声源名称 '{actor_name}' 出错: {e}"

        elif actor_name.startswith("mic_"):
            try:
                idx = int(actor_name.split('_')[1])
                if idx < len(self.current_mic_positions):
                    pos = self.current_mic_positions[idx]
                    info_text = f"选中的麦克风:\n类型: 麦克风\n拾取名称: {actor_name}\n索引: {idx}\n位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m"
                else:
                    info_text = f"选中的麦克风索引 ({idx}) 超出范围。"
            except (IndexError, ValueError) as e:
                info_text = f"解析麦克风名称 '{actor_name}' 出错: {e}"
        else:
            # Could be the room or other elements if they were made pickable
            info_text = f"选中了场景对象: {actor_name}\n(非声源或麦克风)"
            if hasattr(picked_actor, 'center'):
                 info_text += f"\n中心: ({picked_actor.center[0]:.2f}, {picked_actor.center[1]:.2f}, {picked_actor.center[2]:.2f})"

        self.selected_object_info_label.setText(info_text)

# 用于独立运行 GUI 进行测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 