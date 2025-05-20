import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei']

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

    # 绘制声源
    if sources:
        sources_np = np.array(sources)
        ax.scatter(sources_np[:, 0], sources_np[:, 1], sources_np[:, 2], 
                   c='red', marker='o', s=100, label='声源')

    # 绘制麦克风
    if microphones:
        microphones_np = np.array(microphones)
        ax.scatter(microphones_np[:, 0], microphones_np[:, 1], microphones_np[:, 2], 
                   c='blue', marker='x', s=100, label='麦克风')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # 设置坐标轴范围，确保所有元素可见并保持长宽高比例大致正确
    ax.set_xlim([0, lx])
    ax.set_ylim([0, ly])
    ax.set_zlim([0, lz])
    
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
    
    return ax.figure # Return the figure for canvas drawing

# 示例用法
if __name__ == '__main__':
    room_dimensions = [6, 5, 3] # 长, 宽, 高
    example_sources = [[2, 3, 1.5], [1, 1, 1]]
    example_mics = [[4, 4.5, 1.5], [5, 1, 0.5]]
    plot_room_3d(room_dimensions, sources=example_sources, microphones=example_mics)

    plot_room_3d([10,8,4], sources=[[5,4,2]]) # 只有声源
    plot_room_3d([5,5,2.5], microphones=[[1,1,1],[2,3,1.5],[4,2,1]]) # 只有麦克风 