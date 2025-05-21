import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QGridLayout, QGroupBox, QMessageBox,
    QTabWidget, QListWidget, QListWidgetItem, QInputDialog, QDialog, QDialogButtonBox,
    QSizePolicy, QFileDialog, QComboBox, QFormLayout,
    QDoubleSpinBox # Added for numeric input
)
from PySide6.QtCore import Qt, Slot
import json
import os

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

class SineComponentDialog(QDialog):
    def __init__(self, parent=None, freq=440.0, amp=1.0):
        super().__init__(parent)
        self.setWindowTitle("添加/编辑正弦波分量")
        layout = QFormLayout(self)
        
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setSuffix(" Hz")
        self.freq_input.setMinimum(1.0)
        self.freq_input.setMaximum(20000.0)
        self.freq_input.setValue(freq)
        layout.addRow("频率:", self.freq_input)
        
        self.amp_input = QDoubleSpinBox()
        self.amp_input.setDecimals(3)
        self.amp_input.setMinimum(0.0)
        self.amp_input.setMaximum(10.0) # Assuming amplitude is not excessively large
        self.amp_input.setValue(amp)
        layout.addRow("幅度:", self.amp_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_component_data(self):
        return {"freq": self.freq_input.value(), "amp": self.amp_input.value()}

class SourcePropertiesDialog(QDialog):
    def __init__(self, parent=None, source_data=None):
        super().__init__(parent)
        self.setWindowTitle("声源属性")
        self.layout = QVBoxLayout(self)

        self.form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.position_input = QLineEdit()
        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["正弦波组合", "白噪声", "脉冲"])
        
        self.form_layout.addRow("名称:", self.name_input)
        self.form_layout.addRow("位置 (x,y,z):", self.position_input)
        self.form_layout.addRow("信号类型:", self.signal_type_combo)

        # Widget to hold dynamically changing signal parameter UI
        self.signal_params_group = QGroupBox("信号参数") # Use a groupbox for clarity
        self.signal_params_layout = QVBoxLayout() # Layout for the content of the groupbox
        self.signal_params_group.setLayout(self.signal_params_layout)
        self.form_layout.addRow(self.signal_params_group) # Add the groupbox to the form
        
        self.signal_type_combo.currentIndexChanged.connect(self.update_signal_params_ui)

        if source_data: # Populate if editing existing source
            self.name_input.setText(source_data.get("name", ""))
            self.position_input.setText(source_data.get("position_str", ",".join(map(str, source_data.get("position", [])))))
            idx = self.signal_type_combo.findText(source_data.get("signal_type_display", "正弦波组合"))
            if idx != -1: self.signal_type_combo.setCurrentIndex(idx)
            # update_signal_params_ui will be called due to setCurrentIndex if index changes,
            # but we need to ensure it populates with existing data if type is already correct.
            # So, call it explicitly after setting type, and pass existing params.
            self.update_signal_params_ui(self.signal_type_combo.currentIndex(), existing_params=source_data.get("signal_params"))
        else:
            self.update_signal_params_ui(self.signal_type_combo.currentIndex()) # Initial UI for new source

        self.layout.addLayout(self.form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        
    def update_signal_params_ui(self, index, existing_params=None):
        # Clear previous params
        while self.signal_params_layout.count():
            child = self.signal_params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        selected_type = self.signal_type_combo.currentText()
        self.current_signal_type_key = selected_type # Store for get_source_data

        if selected_type == "正弦波组合":
            # TODO: Implement UI for sine combo: list of freq/amp pairs
            self.sine_combo_widget = QWidget() # Placeholder parent widget
            self.sine_combo_layout = QVBoxLayout(self.sine_combo_widget)
            
            self.sine_components_list = QListWidget()
            self.sine_combo_layout.addWidget(self.sine_components_list)
            
            buttons_layout = QHBoxLayout()
            self.add_sine_button = QPushButton("添加分量")
            self.add_sine_button.clicked.connect(self.add_sine_component)
            self.remove_sine_button = QPushButton("移除选中分量")
            self.remove_sine_button.clicked.connect(self.remove_sine_component)
            buttons_layout.addWidget(self.add_sine_button)
            buttons_layout.addWidget(self.remove_sine_button)
            self.sine_combo_layout.addLayout(buttons_layout)
            
            if existing_params and "components" in existing_params:
                for comp in existing_params["components"]:
                    item_text = f"频率: {comp['freq']:.1f} Hz, 幅度: {comp['amp']:.3f}"
                    list_item = QListWidgetItem(item_text)
                    # Store the actual data in the item for easier retrieval/editing
                    list_item.setData(Qt.UserRole, comp) 
                    self.sine_components_list.addItem(list_item)
            
            self.signal_params_layout.addWidget(self.sine_combo_widget)

        elif selected_type == "白噪声":
            self.signal_params_layout.addWidget(QLabel("无特定参数"))
            # No specific controls needed for white noise based on current requirements

        elif selected_type == "脉冲":
            self.pulse_params_widget = QWidget()
            pulse_form = QFormLayout(self.pulse_params_widget)
            self.pulse_width_input = QDoubleSpinBox()
            self.pulse_width_input.setSuffix(" s")
            self.pulse_width_input.setDecimals(4)
            self.pulse_width_input.setMinimum(0.0001)
            self.pulse_width_input.setMaximum(10.0)
            self.pulse_width_input.setSingleStep(0.001)
            self.pulse_width_input.setValue(existing_params.get("width", 0.001) if existing_params else 0.001)
            pulse_form.addRow("脉冲宽度:", self.pulse_width_input)
            self.signal_params_layout.addWidget(self.pulse_params_widget)
        
        # Ensure the group box title reflects the current selection
        self.signal_params_group.setTitle(f"{selected_type} - 参数")

    @Slot()
    def add_sine_component(self):
        dialog = SineComponentDialog(self)
        if dialog.exec() == QDialog.Accepted:
            comp_data = dialog.get_component_data()
            item_text = f"频率: {comp_data['freq']:.1f} Hz, 幅度: {comp_data['amp']:.3f}"
            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, comp_data)
            self.sine_components_list.addItem(list_item)

    @Slot()
    def remove_sine_component(self):
        selected_items = self.sine_components_list.selectedItems()
        if not selected_items: return
        for item in selected_items:
            self.sine_components_list.takeItem(self.sine_components_list.row(item))

    def get_source_data(self):
        pos_str = self.position_input.text()
        name = self.name_input.text() or f"Source{np.random.randint(1000,9999)}" 
        signal_type_display = self.signal_type_combo.currentText()
        # Map display name to an internal key if needed, for now, they are the same
        signal_type_key = self.current_signal_type_key 

        params = {}
        if signal_type_key == "正弦波组合":
            # TODO: Collect from dynamic UI for sine components
            components = []
            if hasattr(self, 'sine_components_list'):
                for i in range(self.sine_components_list.count()):
                    list_item = self.sine_components_list.item(i)
                    comp_data = list_item.data(Qt.UserRole)
                    if comp_data: # Make sure data exists
                        components.append(comp_data)
            params["components"] = components
            # params["description"] = "正弦波组合参数待实现"
        elif signal_type_key == "白噪声":
            pass # No specific params to collect
        elif signal_type_key == "脉冲":
            if hasattr(self, 'pulse_width_input'): # Check if UI element was created
                params["width"] = self.pulse_width_input.value()
            else: # Should not happen if UI updated correctly
                params["width"] = 0.001 # Fallback default

        return {
            "name": name,
            "position_str": pos_str, 
            "signal_type_display": signal_type_display,
            "signal_type": signal_type_key, 
            "signal_params": params
        }

class MicrophonePropertiesDialog(QDialog):
    def __init__(self, parent=None, mic_data=None):
        super().__init__(parent)
        self.setWindowTitle("麦克风属性")
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.position_input = QLineEdit()
        self.sensitivity_input = QDoubleSpinBox()
        self.sensitivity_input.setDecimals(3)
        self.sensitivity_input.setMinimum(0.001)
        self.sensitivity_input.setMaximum(100.0) # Arbitrary practical max
        self.sensitivity_input.setValue(1.0)
        
        self.noise_std_input = QDoubleSpinBox()
        self.noise_std_input.setDecimals(5)
        self.noise_std_input.setMinimum(0.0)
        self.noise_std_input.setMaximum(1.0)
        self.noise_std_input.setValue(0.001)
        self.noise_std_input.setSingleStep(0.0001)

        self.freq_response_type_combo = QComboBox()
        self.freq_response_type_combo.addItems(["无", "低通", "高通", "带通"])

        self.form_layout.addRow("名称:", self.name_input)
        self.form_layout.addRow("位置 (x,y,z):", self.position_input)
        self.form_layout.addRow("灵敏度:", self.sensitivity_input)
        self.form_layout.addRow("自噪声标准差:", self.noise_std_input)
        self.form_layout.addRow("频响类型:", self.freq_response_type_combo)

        self.freq_params_group = QGroupBox("频响参数")
        self.freq_params_layout = QFormLayout() # Using QFormLayout for param pairs
        self.freq_params_group.setLayout(self.freq_params_layout)
        self.form_layout.addRow(self.freq_params_group)

        self.freq_response_type_combo.currentIndexChanged.connect(self.update_freq_params_ui)

        if mic_data:
            self.name_input.setText(mic_data.get("name", ""))
            self.position_input.setText(mic_data.get("position_str", ",".join(map(str, mic_data.get("position", [])))))
            self.sensitivity_input.setValue(mic_data.get("sensitivity", 1.0))
            self.noise_std_input.setValue(mic_data.get("noise_std", 0.001))
            idx = self.freq_response_type_combo.findText(mic_data.get("freq_response_type_display", "无"))
            if idx != -1: self.freq_response_type_combo.setCurrentIndex(idx)
            self.update_freq_params_ui(self.freq_response_type_combo.currentIndex(), existing_params=mic_data.get("freq_response_params"))
        else:
            self.update_freq_params_ui(self.freq_response_type_combo.currentIndex())

        self.layout.addLayout(self.form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def update_freq_params_ui(self, index, existing_params=None):
        while self.freq_params_layout.rowCount() > 0:
            self.freq_params_layout.removeRow(0)
        
        selected_type = self.freq_response_type_combo.currentText()
        self.current_freq_response_type_key = selected_type # Store for get_mic_data
        self.freq_params_group.setTitle(f"{selected_type} - 参数")

        if selected_type == "低通" or selected_type == "高通":
            self.cutoff_freq_input = QDoubleSpinBox()
            self.cutoff_freq_input.setSuffix(" Hz")
            self.cutoff_freq_input.setMinimum(10.0)
            self.cutoff_freq_input.setMaximum(20000.0) # Typical audio range
            self.cutoff_freq_input.setValue(existing_params.get("cutoff", 1000.0) if existing_params else 1000.0)
            self.freq_params_layout.addRow("截止频率:", self.cutoff_freq_input)
            self.freq_params_group.setVisible(True)
        elif selected_type == "带通":
            self.low_cutoff_freq_input = QDoubleSpinBox()
            self.low_cutoff_freq_input.setSuffix(" Hz")
            self.low_cutoff_freq_input.setMinimum(10.0)
            self.low_cutoff_freq_input.setMaximum(19990.0)
            self.low_cutoff_freq_input.setValue(existing_params.get("low_cutoff", 500.0) if existing_params else 500.0)
            
            self.high_cutoff_freq_input = QDoubleSpinBox()
            self.high_cutoff_freq_input.setSuffix(" Hz")
            self.high_cutoff_freq_input.setMinimum(20.0)
            self.high_cutoff_freq_input.setMaximum(20000.0)
            self.high_cutoff_freq_input.setValue(existing_params.get("high_cutoff", 2000.0) if existing_params else 2000.0)
            
            self.freq_params_layout.addRow("低截止频率:", self.low_cutoff_freq_input)
            self.freq_params_layout.addRow("高截止频率:", self.high_cutoff_freq_input)
            self.freq_params_group.setVisible(True)
        else: # "无"
            self.freq_params_group.setVisible(False)

    def get_mic_data(self):
        params = {}
        if self.current_freq_response_type_key == "低通" or self.current_freq_response_type_key == "高通":
            if hasattr(self, 'cutoff_freq_input'):
                params["cutoff"] = self.cutoff_freq_input.value()
        elif self.current_freq_response_type_key == "带通":
            if hasattr(self, 'low_cutoff_freq_input') and hasattr(self, 'high_cutoff_freq_input'):
                params["low_cutoff"] = self.low_cutoff_freq_input.value()
                params["high_cutoff"] = self.high_cutoff_freq_input.value()
                if params["low_cutoff"] >= params["high_cutoff"]:
                    # Basic validation, could be more robust
                    raise ValueError("带通滤波器的低截止频率必须小于高截止频率。")
        return {
            "name": self.name_input.text() or f"Mic{np.random.randint(1000,9999)}",
            "position_str": self.position_input.text(),
            "sensitivity": self.sensitivity_input.value(),
            "noise_std": self.noise_std_input.value(),
            "freq_response_type_display": self.freq_response_type_combo.currentText(),
            "freq_response_type": self.current_freq_response_type_key,
            "freq_response_params": params
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式声学仿真工具 v0.4")
        self.setGeometry(100, 100, 1300, 850)

        # 先初始化数据结构
        self.sources_data = []
        self.mics_data = []

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
        self.edit_source_button = QPushButton("编辑选中声源")
        self.edit_source_button.clicked.connect(self.edit_selected_source)
        self.remove_source_button = QPushButton("移除选中声源")
        self.remove_source_button.clicked.connect(self.remove_source)
        sources_buttons_layout.addWidget(self.add_source_button)
        sources_buttons_layout.addWidget(self.edit_source_button)
        sources_buttons_layout.addWidget(self.remove_source_button)
        sources_layout.addLayout(sources_buttons_layout)
        sources_group.setLayout(sources_layout)
        left_panel_layout.addWidget(sources_group)
        # Add default source
        default_source = {
            "name": "Default Source",
            "position": [2.0, 3.0, 1.5],
            "position_str": "2,3,1.5",
            "signal_type_display": "正弦波组合",
            "signal_type": "正弦波组合",
            "signal_params": {"components": [{"freq": 440, "amp": 1.0}]}
        }
        self.sources_data.append(default_source)
        display_text = f"{default_source['name']}: {default_source['position_str']} ({default_source['signal_type_display']})"
        self.sources_list_widget.addItem(display_text)

        # Microphones Group
        mics_group = QGroupBox("麦克风管理")
        mics_layout = QVBoxLayout()
        self.mics_list_widget = QListWidget()
        self.mics_list_widget.setFixedHeight(100)
        mics_layout.addWidget(self.mics_list_widget)
        mics_buttons_layout = QHBoxLayout()
        self.add_mic_button = QPushButton("添加麦克风")
        self.add_mic_button.clicked.connect(self.add_mic)
        self.edit_mic_button = QPushButton("编辑选中麦克风")
        self.edit_mic_button.clicked.connect(self.edit_selected_mic)
        self.remove_mic_button = QPushButton("移除选中麦克风")
        self.remove_mic_button.clicked.connect(self.remove_mic)
        mics_buttons_layout.addWidget(self.add_mic_button)
        mics_buttons_layout.addWidget(self.edit_mic_button)
        mics_buttons_layout.addWidget(self.remove_mic_button)
        mics_layout.addLayout(mics_buttons_layout)
        mics_group.setLayout(mics_layout)
        left_panel_layout.addWidget(mics_group)
        # Add default microphones
        default_mic = {
            "name": "Default Mic",
            "position": [4.0, 2.5, 1.5],
            "position_str": "4,2.5,1.5",
            "sensitivity": 1.0,
            "noise_std": 0.001,
            "freq_response_type_display": "无",
            "freq_response_type": "无",
            "freq_response_params": {}
        }
        self.mics_data.append(default_mic)
        display_text = f"{default_mic['name']}: {default_mic['position_str']} ({default_mic['freq_response_type_display']})"
        self.mics_list_widget.addItem(display_text)

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
        # 新增保存/加载配置按钮
        self.save_config_button = QPushButton("保存配置")
        self.save_config_button.clicked.connect(self.save_config)
        control_layout_v.addWidget(self.run_button)
        control_layout_v.addWidget(self.save_config_button)
        self.load_config_button = QPushButton("加载配置")
        self.load_config_button.clicked.connect(self.load_config)
        control_layout_v.addWidget(self.load_config_button)
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
            self.pv_plotter.add_camera_orientation_widget(True) # 添加坐标轴方向指示
            self.pv_plotter.add_bounding_box() # 更正: 添加边界框控制应为add_bounding_box()
            self.pv_plotter.show_grid() # 显示网格，如果需要的话

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
        dialog = SourcePropertiesDialog(self)
        if dialog.exec() == QDialog.Accepted:
            new_source_data = dialog.get_source_data()
            try:
                # Validate position format before adding
                pos_vec = self.parse_vector_input(new_source_data["position_str"], 3)
                new_source_data["position"] = pos_vec # Store parsed position
                
                # For now, list widget shows name and position
                display_text = f"{new_source_data['name']}: {new_source_data['position_str']} ({new_source_data['signal_type_display']})"
                self.sources_list_widget.addItem(display_text)
                self.sources_data.append(new_source_data)

            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"无效的声源位置格式: {new_source_data['position_str']}\n{e}\n请输入类似 '1,2,3' 的格式。")
            except Exception as e:
                 QMessageBox.critical(self, "添加失败", f"添加声源时出错: {e}")

    @Slot()
    def edit_selected_source(self):
        selected_items = self.sources_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要编辑的声源。")
            return
        if len(selected_items) > 1:
            QMessageBox.warning(self, "操作无效", "一次只能编辑一个声源。")
            return
        
        current_item = selected_items[0]
        current_row = self.sources_list_widget.row(current_item)
        
        if current_row < 0 or current_row >= len(self.sources_data):
            QMessageBox.critical(self, "错误", "选中项与内部数据不匹配，请重试。")
            return
            
        existing_source_data = self.sources_data[current_row]
        
        # Ensure position_str is present for the dialog, if only position list exists
        if "position" in existing_source_data and "position_str" not in existing_source_data:
            existing_source_data["position_str"] = ",".join(map(str, existing_source_data["position"]))

        dialog = SourcePropertiesDialog(self, source_data=existing_source_data)
        if dialog.exec() == QDialog.Accepted:
            updated_source_data = dialog.get_source_data()
            try:
                pos_vec = self.parse_vector_input(updated_source_data["position_str"], 3)
                updated_source_data["position"] = pos_vec
                
                # Update data list
                self.sources_data[current_row] = updated_source_data
                
                # Update QListWidget item text
                display_text = f"{updated_source_data['name']}: {updated_source_data['position_str']} ({updated_source_data['signal_type_display']})"
                current_item.setText(display_text)
                
                QMessageBox.information(self, "成功", f"声源 '{updated_source_data['name']}' 已更新。")
                
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"无效的声源位置格式: {updated_source_data['position_str']}\n{e}")
            except Exception as e:
                QMessageBox.critical(self, "更新失败", f"更新声源时出错: {e}")

    def remove_source(self):
        selected_items = self.sources_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要移除的声源。")
            return
        for item in selected_items:
            self.sources_list_widget.takeItem(self.sources_list_widget.row(item))
        if self.sources_list_widget.count() == 0: # Ensure at least one source if all are removed
            QMessageBox.information(self, "提示", "至少需要一个声源。已自动添加默认声源。")
            # Add a default source to self.sources_data and list_widget if needed
            default_source = {
                "name": "Default Source",
                "position": [2.0, 3.0, 1.5],
                "position_str": "2,3,1.5",
                "signal_type_display": "正弦波组合",
                "signal_type": "正弦波组合",
                "signal_params": {"components": [{"freq": 440, "amp": 1.0}]}
            }
            self.sources_data.append(default_source)
            display_text = f"{default_source['name']}: {default_source['position_str']} ({default_source['signal_type_display']})"
            self.sources_list_widget.addItem(display_text)

    def add_mic(self):
        dialog = MicrophonePropertiesDialog(self)
        if dialog.exec() == QDialog.Accepted:
            try:
                new_mic_data = dialog.get_mic_data()
                pos_vec = self.parse_vector_input(new_mic_data["position_str"], 3)
                new_mic_data["position"] = pos_vec
                
                display_text = f"{new_mic_data['name']}: {new_mic_data['position_str']} ({new_mic_data['freq_response_type_display']})"
                self.mics_list_widget.addItem(display_text)
                self.mics_data.append(new_mic_data)
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"麦克风参数错误: {e}")
            except Exception as e:
                QMessageBox.critical(self, "添加失败", f"添加麦克风时出错: {e}")

    @Slot()
    def edit_selected_mic(self):
        selected_items = self.mics_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要编辑的麦克风。")
            return
        if len(selected_items) > 1:
            QMessageBox.warning(self, "操作无效", "一次只能编辑一个麦克风。")
            return
        
        current_item = selected_items[0]
        current_row = self.mics_list_widget.row(current_item)
        
        if current_row < 0 or current_row >= len(self.mics_data):
            QMessageBox.critical(self, "错误", "选中项与内部数据不匹配，请重试。")
            return
            
        existing_mic_data = self.mics_data[current_row]
        if "position" in existing_mic_data and "position_str" not in existing_mic_data:
            existing_mic_data["position_str"] = ",".join(map(str, existing_mic_data["position"]))

        dialog = MicrophonePropertiesDialog(self, mic_data=existing_mic_data)
        if dialog.exec() == QDialog.Accepted:
            try:
                updated_mic_data = dialog.get_mic_data()
                pos_vec = self.parse_vector_input(updated_mic_data["position_str"], 3)
                updated_mic_data["position"] = pos_vec
                
                self.mics_data[current_row] = updated_mic_data
                display_text = f"{updated_mic_data['name']}: {updated_mic_data['position_str']} ({updated_mic_data['freq_response_type_display']})"
                current_item.setText(display_text)
                QMessageBox.information(self, "成功", f"麦克风 '{updated_mic_data['name']}' 已更新。")
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"麦克风参数错误: {e}")
            except Exception as e:
                QMessageBox.critical(self, "更新失败", f"更新麦克风时出错: {e}")

    def remove_mic(self):
        selected_items = self.mics_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "操作无效", "请先选择一个要移除的麦克风。")
            return
        # Remove from both list widget and data list carefully
        rows_to_remove = sorted([self.mics_list_widget.row(item) for item in selected_items], reverse=True)
        for row in rows_to_remove:
            self.mics_list_widget.takeItem(row)
            if row < len(self.mics_data): # Check bounds before popping
                self.mics_data.pop(row)
            
        if self.mics_list_widget.count() == 0: 
            QMessageBox.information(self, "提示", "至少需要一个麦克风。已自动添加默认麦克风。")
            default_mic = {
                "name": "Default Mic",
                "position": [4.0, 2.5, 1.5],
                "position_str": "4,2.5,1.5",
                "sensitivity": 1.0,
                "noise_std": 0.001,
                "freq_response_type_display": "无",
                "freq_response_type": "无",
                "freq_response_params": {}
            }
            self.mics_data.append(default_mic)
            display_text = f"{default_mic['name']}: {default_mic['position_str']} ({default_mic['freq_response_type_display']})"
            self.mics_list_widget.addItem(display_text)

    def run_simulation_and_update_plots(self):
        try:
            room_dims_val = self.parse_vector_input(self.room_dims_input.text(), 3)
            rt60_val = float(self.rt60_input.text())
            
            source_positions_val = [s_data["position"] for s_data in self.sources_data if "position" in s_data]
            mic_positions_val = [m_data["position"] for m_data in self.mics_data if "position" in m_data]
            
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

            if not self.sources_data:
                 QMessageBox.warning(self, "配置错误", "没有定义声源。")
                 return
            
            # Create SoundSource instances from self.sources_data
            source_objects = []
            for src_data in self.sources_data:
                source_objects.append(SoundSource(
                    position=np.array(src_data["position"]),
                    name=src_data.get("name", f"Source_DefaultName"),
                    signal_type=src_data.get("signal_type", "白噪声"),
                    signal_params=src_data.get("signal_params", {})
                ))
            
            if not source_objects: # Should be redundant if self.sources_data is checked
                QMessageBox.warning(self, "配置错误", "未能创建声源对象。")
                return

            mic_objects = []
            for i, m_data in enumerate(self.mics_data):
                mic_objects.append(Microphone(
                    position=np.array(m_data["position"]),
                    name=m_data.get("name", f"Mic{i+1}"),
                    sensitivity=m_data.get("sensitivity", 1.0),
                    self_noise_std=m_data.get("noise_std", 0.01),
                    freq_response_type=m_data.get("freq_response_type"),
                    cutoff_freqs=m_data.get("freq_response_params") # Pass the whole params dict
                ))

            self.recorded_signals, self.simulation_room = simulate_with_pyroomacoustics(
                room_dim=room_dims_val, 
                source_objects=source_objects, # Pass list of source objects
                mic_objects=mic_objects, 
                duration=self.current_duration, 
                rt60=rt60_val
            )
            # For multi-source, ground_truth_signal needs clarification.
            # Using the first source's signal as a reference for now.
            if source_objects:
                self.ground_truth_signal = source_objects[0].get_signal(self.current_duration)
            else:
                self.ground_truth_signal = None # Should not happen if checks above are done
            
            # Update 3D Plot with PyVista
            if self.pv_plotter is not None and PYVISTA_AVAILABLE:
                create_pyvista_scene(self.pv_plotter, 
                                     room_dim=room_dims_val, 
                                     sources=source_positions_val, # This is a list of positions
                                     microphones=mic_positions_val)
                # self.current_source_positions = source_positions_val # Replaced by self.sources_data
                # self.current_mic_positions = mic_positions_val     # Replaced by self.mics_data (later)
                
                # Setup picking callback for PyVista QtInteractor
                # The enable_actor_picking might be slightly different or might work directly
                # on QtInteractor as it inherits from BasePlotter.
                if hasattr(self.pv_plotter, 'track_click_position'):
                    self.pv_plotter.track_click_position(callback=self.on_pick_pyvista)
                else:
                    print("QtInteractor does not have track_click_position directly, check documentation or alternative.")
                
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
            # With multiple sources, room.rir is a list of RIRs (one per source).
            # Plot RIR from first source to first mic for now.
            if self.simulation_room.rir and len(self.simulation_room.rir) > 0 and \
               self.simulation_room.rir[0] and len(self.simulation_room.rir[0]) > 0 and \
               self.simulation_room.rir[0][0] is not None:
                plot_rir_embed(self.ax_rir, self.simulation_room.rir[0][0], SAMPLING_RATE, 
                               title=f"RIR ({mic_objects[0].name} vs {source_objects[0].name})")
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

    def on_pick_pyvista(self, position):
        info_text = "点击了3D场景中的空白区域。"
        if position is None:
            self.selected_object_info_label.setText(info_text)
            return

        # Determine if it's a source or microphone based on the position
        for s_data in self.sources_data:
            if np.linalg.norm(np.array(s_data["position"]) - position) < 0.1:
                # Assuming a tolerance of 0.1 meters for proximity
                info_text = f"选中的声源:\n名称: {s_data.get('name')}\n位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m\n信号: {s_data.get('signal_type_display')}"
                self.selected_object_info_label.setText(info_text)
                return
        for m_data in self.mics_data:
            if np.linalg.norm(np.array(m_data["position"]) - position) < 0.1:
                info_text = f"选中的麦克风:\n名称: {m_data.get('name')}\n位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m\n灵敏度: {m_data.get('sensitivity')}\n频响: {m_data.get('freq_response_type_display')}"
                self.selected_object_info_label.setText(info_text)
                return
        # Could be the room or other elements if they were made pickable
        info_text = f"选中了场景对象: {position}\n(非声源或麦克风)"
        if hasattr(self.pv_plotter, 'center'):
            info_text += f"\n中心: ({self.pv_plotter.center[0]:.2f}, {self.pv_plotter.center[1]:.2f}, {self.pv_plotter.center[2]:.2f})"
        self.selected_object_info_label.setText(info_text)

    def save_config(self):
        config = {
            "room_dim": self.room_dims_input.text(),
            "rt60": self.rt60_input.text(),
            "duration": self.duration_input.text(),
            "sources_data": self.sources_data, 
            "mics_data": self.mics_data # Save the full mic data objects
        }
        config["room_dim"] = [float(x) for x in config["room_dim"].split(",")]
        config["rt60"] = float(config["rt60"])
        config["duration"] = float(config["duration"])
        # 默认保存目录
        default_config_dir = os.path.join(os.path.dirname(__file__), '../../configs')
        os.makedirs(default_config_dir, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "保存配置", default_config_dir, "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "成功", f"配置已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存配置时出错: {e}")

    def load_config(self):
        # 默认加载目录
        default_config_dir = os.path.join(os.path.dirname(__file__), '../../configs')
        os.makedirs(default_config_dir, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "加载配置", default_config_dir, "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.room_dims_input.setText(",".join(str(x) for x in config["room_dim"]))
                self.rt60_input.setText(str(config["rt60"]))
                self.duration_input.setText(str(config["duration"]))
                self.sources_list_widget.clear()
                self.sources_data = config.get("sources_data", [])
                for s_data in self.sources_data:
                    if "position" in s_data and "position_str" not in s_data:
                        s_data["position_str"] = ",".join(map(str, s_data["position"]))
                    display_text = f"{s_data.get('name', 'Source')}: {s_data.get('position_str', 'N/A')} ({s_data.get('signal_type_display', 'N/A')})"
                    self.sources_list_widget.addItem(display_text)
                self.mics_list_widget.clear()
                self.mics_data = config.get("mics_data", [])
                for m_data in self.mics_data:
                    if "position" in m_data and "position_str" not in m_data:
                        m_data["position_str"] = ",".join(map(str, m_data["position"]))
                    display_text = f"{m_data.get('name', 'Mic')}: {m_data.get('position_str', 'N/A')} ({m_data.get('freq_response_type_display', 'N/A')})"
                    self.mics_list_widget.addItem(display_text)
                QMessageBox.information(self, "成功", f"配置已加载: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载配置时出错: {e}")

# 用于独立运行 GUI 进行测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 