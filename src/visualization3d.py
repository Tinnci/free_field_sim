import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei']

# PyVista imports
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    BackgroundPlotter = None # Placeholder if not available

def plot_room_3d(room_dim, sources=None, microphones=None, title="3D声学场景", ax=None):
    """
    使用 Matplotlib 3D 绘制房间、声源和麦克风的简单3D视图。

    :param room_dim: 房间尺寸 [length, width, height]
    :param sources: 声源位置列表，每个元素是 [x,y,z]
    :param microphones: 麦克风位置列表，每个元素是 [x,y,z]
    :param title: 图像标题
    :param ax: Matplotlib 3D Axes object to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        ax.clear() # Clear the axes if it's being reused
        show_plot = False

    # 绘制房间边界 (线框)
    lx, ly, lz = room_dim
    # 定义房间的8个顶点
    vertices = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],  # 底部4个点
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]   # 顶部4个点
    ])
    # 定义构成房间线框的边
    edges = [
        [vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[3]], [vertices[3], vertices[0]], # 底边
        [vertices[4], vertices[5]], [vertices[5], vertices[6]], [vertices[6], vertices[7]], [vertices[7], vertices[4]], # 顶边
        [vertices[0], vertices[4]], [vertices[1], vertices[5]], [vertices[2], vertices[6]], [vertices[3], vertices[7]]  # 垂直边
    ]
    for edge in edges:
        ax.plot3D(*zip(*edge), color="gray")

    ax.set_title(title)
    
    # 设置坐标轴范围，确保所有元素可见并保持长宽高比例大致正确
    # 同时让用户能够通过鼠标滚轮缩放
    ax.set_xlim([0, lx])
    ax.set_ylim([0, ly])
    ax.set_zlim([0, lz])
    
    # 存储绘制的艺术家对象，以便外部访问
    plotted_elements = {}

    # 绘制声源
    if sources:
        sources_np = np.array(sources)
        plotted_elements['sources'] = ax.scatter(sources_np[:, 0], sources_np[:, 1], sources_np[:, 2], 
                   c='red', marker='o', s=100, label='声源', picker=True, pickradius=5) # Enable picking

    # 绘制麦克风
    if microphones:
        microphones_np = np.array(microphones)
        plotted_elements['microphones'] = ax.scatter(microphones_np[:, 0], microphones_np[:, 1], microphones_np[:, 2], 
                   c='blue', marker='x', s=100, label='麦克风', picker=True, pickradius=5) # Enable picking

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 确保图例只显示一次或在需要时更新
    if ax.get_legend() is None or not ax.get_legend().get_texts():
        if sources or microphones:
            ax.legend()
    else:
        if sources or microphones: # Potentially update legend if items change
            ax.legend()
        else:
            ax.get_legend().remove()
            
    ax.grid(True)

    if show_plot:
        plt.show()
    
    return ax.figure, plotted_elements # Return the figure and plotted elements

# 示例用法
if __name__ == '__main__':
    room_dimensions = [6, 5, 3] # 长, 宽, 高
    example_sources = [[2, 3, 1.5], [1, 1, 1]]
    example_mics = [[4, 4.5, 1.5], [5, 1, 0.5]]
    plot_room_3d(room_dimensions, sources=example_sources, microphones=example_mics)

    plot_room_3d([10,8,4], sources=[[5,4,2]]) # 只有声源
    plot_room_3d([5,5,2.5], microphones=[[1,1,1],[2,3,1.5],[4,2,1]]) # 只有麦克风

# --- PyVista Implementation ---

def create_pyvista_scene(plotter, room_dim, sources=None, microphones=None):
    """
    使用 PyVista 在给定的 plotter 对象中创建和绘制3D声学场景。

    :param plotter: pyvistaqt.BackgroundPlotter 实例。
    :param room_dim: 房间尺寸 [length, width, height]
    :param sources: 声源位置列表，每个元素是 [x,y,z]
    :param microphones: 麦克风位置列表，每个元素是 [x,y,z]
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista is not installed. Cannot create PyVista scene.")
        return

    plotter.clear_actors() # Clear previous actors

    # 绘制房间 (线框立方体)
    lx, ly, lz = room_dim
    # 中心点和边界来定义立方体
    center = [lx/2, ly/2, lz/2]
    bounds = [0, lx, 0, ly, 0, lz]
    room_mesh = pv.Cube(center=center, x_length=lx, y_length=ly, z_length=lz)
    # plotter.add_mesh(room_mesh, style='wireframe', color='gray', line_width=2, label="Room")
    
    # 或者使用明确的边来绘制房间轮廓，更像之前的matplotlib版本
    # 定义房间的8个顶点
    vertices = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],  # 底部4个点
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]   # 顶部4个点
    ])
    # 定义构成房间线框的边 (索引对)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 底边
        [4, 5], [5, 6], [6, 7], [7, 4], # 顶边
        [0, 4], [1, 5], [2, 6], [3, 7]  # 垂直边
    ]
    for edge in edges:
        plotter.add_lines(np.array([vertices[edge[0]], vertices[edge[1]]]), color="gray", width=3)


    # 绘制声源
    if sources:
        for i, src_pos in enumerate(sources):
            source_sphere = pv.Sphere(center=src_pos, radius=0.1) # 可调整半径
            # 为拾取添加名称标签或自定义属性
            plotter.add_mesh(source_sphere, color='red', label=f'Source {i}', name=f'source_{i}')

    # 绘制麦克风
    if microphones:
        for i, mic_pos in enumerate(microphones):
            mic_sphere = pv.Sphere(center=mic_pos, radius=0.08) # 可调整半径
            # 使用不同形状或仅颜色区分
            plotter.add_mesh(mic_sphere, color='blue', label=f'Microphone {i}', name=f'mic_{i}')

    # 设置相机视角等
    plotter.camera_position = 'iso' # 等轴测视图
    plotter.camera.azimuth = -45
    plotter.camera.elevation = 20
    # plotter.enable_zoom_scaling() # QtInteractor 通常默认支持滚轮缩放，或者通过交互样式控制
                                  # AttributeError: QtInteractor has no attribute named enable_zoom_scaling. Did you mean: 'enable_zoom_style'?
    # plotter.show_grid(False)
    # plotter.show_axes() # 可以选择显示坐标轴

    # 添加背景颜色
    plotter.set_background('lightgrey') # 例如浅灰色


if __name__ == '__main__':
    # Matplotlib example (existing)
    # room_dimensions_mpl = [6, 5, 3]
    # example_sources_mpl = [[2, 3, 1.5], [1, 1, 1]]
    # example_mics_mpl = [[4, 4.5, 1.5], [5, 1, 0.5]]
    # plot_room_3d(room_dimensions_mpl, sources=example_sources_mpl, microphones=example_mics_mpl)

    # PyVista example (for testing this script directly)
    if PYVISTA_AVAILABLE:
        plotter_test = BackgroundPlotter(show=True, window_size=(800,600)) # Create a plotter
        room_dims_pv = [7, 6, 3.5]
        sources_pv = [[1, 1, 1], [5, 4, 1.5]]
        mics_pv = [[3, 3, 1], [6, 1, 0.5]]
        create_pyvista_scene(plotter_test, room_dims_pv, sources=sources_pv, microphones=mics_pv)
        # For standalone test, plotter.app.exec_() might be needed if not showing window immediately
        # or if BackgroundPlotter is not set to show.
        # In Qt context, this direct exec is not needed.
        print("PyVista test scene created. If a window appeared, you can close it to continue.")
        # To keep window open for interaction in a script:
        # import sys
        # if sys.flags.interactive:
        #     pv.plotting.show() # Or some form of plotter.app.exec_()
        # else:
        #     plotter_test.app.exec_() # Or manage via the Qt app in main_window
    else:
        print("PyVista not found, skipping PyVista direct test.") 