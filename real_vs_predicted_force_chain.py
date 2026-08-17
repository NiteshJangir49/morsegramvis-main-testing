import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSlider, QCheckBox, QSplitter, QFrame, QDialog
)
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Matplotlib integration for Figure 3(d) Matrix Rendering
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


# ==============================================================================
# DATA LOADING UTILITY (Assembly 58, Timestep 80)
# ==============================================================================
def load_real_and_predicted_data(assembly_id=58, timestep=80):
    """Loads 3D positions, actual NPMNCF (Vol A), and predicted NPMNCF (Vol B)."""
    
    actual_dir = os.path.join("real_dataset", "actual_data")
    predicted_dir = os.path.join("real_dataset", "predicted_data")

    def resolve_path(directory, filename):
        path_in_dir = os.path.join(directory, filename)
        if os.path.exists(path_in_dir):
            return path_in_dir
        if os.path.exists(filename):
            return filename
        return path_in_dir

    pos_x_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_ball_pos_x.tab")
    pos_y_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_ball_pos_y.tab")
    pos_z_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_ball_pos_z.tab")
    actual_npmncf_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_ball_NPMNCF.tab")

    pred_npmncf_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_npmncf.csv")
    if not os.path.exists(pred_npmncf_path):
        pred_npmncf_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_predicted_npmncf.csv")

    # 1. Load 3D Coordinates (Actual Data)
    df_x = pd.read_csv(pos_x_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'x'])
    df_y = pd.read_csv(pos_y_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'y'])
    df_z = pd.read_csv(pos_z_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'z'])

    df_x['ball_id'] = df_x['ball_id'].astype(int)
    df_y['ball_id'] = df_y['ball_id'].astype(int)
    df_z['ball_id'] = df_z['ball_id'].astype(int)

    df_coords = df_x.merge(df_y, on='ball_id').merge(df_z, on='ball_id')

    # 2. Load Actual NPMNCF (Volume A) and normalize per paper standard
    df_actual = pd.read_csv(actual_npmncf_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'npmncf_raw'])
    df_actual['ball_id'] = pd.to_numeric(df_actual['ball_id'], errors='coerce')
    df_actual.dropna(subset=['ball_id'], inplace=True)
    df_actual['ball_id'] = df_actual['ball_id'].astype(int)

    mean_force = df_actual['npmncf_raw'].mean()
    df_actual['node_weight_actual'] = df_actual['npmncf_raw'] / mean_force if mean_force > 0 else df_actual['npmncf_raw']

    # 3. Load GNN Predicted NPMNCF (Volume B)
    df_pred = pd.read_csv(pred_npmncf_path)

    # Merge Node Data
    df_nodes = df_pred.merge(df_actual[['ball_id', 'node_weight_actual']], on='ball_id', how='inner')
    df_nodes = df_nodes.merge(df_coords, on='ball_id', how='inner')

    ball_ids = df_nodes['ball_id'].values
    num_nodes = len(ball_ids)
    id_to_idx = {bid: idx for idx, bid in enumerate(ball_ids)}

    # Rescale coordinates for VTK 3D space
    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float32) * 100.0

    node_weights_a = df_nodes['node_weight_actual'].values.astype(np.float32)
    node_weights_b = df_nodes['predicted_npmncf'].values.astype(np.float32)

    # 4. Load Contact Connectivity (Fixed Original Contacts)
    end1_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end1.tab")
    end2_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end2.tab")
    pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_contact_force.csv")
    if not os.path.exists(pred_contact_path):
        pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_predicted_contact_force.csv")
    contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_contact_pairs.csv")
    if not os.path.exists(contact_pairs_path):
        contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_contact_pairs.csv")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    weights_a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    weights_b = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if os.path.exists(end1_path) and os.path.exists(end2_path):
        e1 = pd.read_csv(end1_path, sep=r'\s+', header=None, skiprows=2)
        e2 = pd.read_csv(end2_path, sep=r'\s+', header=None, skiprows=2)
        e1_ids = pd.to_numeric(e1.iloc[:, 1], errors='coerce').dropna().astype(int)
        e2_ids = pd.to_numeric(e2.iloc[:, 1], errors='coerce').dropna().astype(int)
        for u_id, v_id in zip(e1_ids, e2_ids):
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])
    elif os.path.exists(pred_contact_path):
        df_contacts = pd.read_csv(pred_contact_path)
        for _, row in df_contacts.iterrows():
            u_id, v_id = int(row['sender_id']), int(row['receiver_id'])
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])
    elif os.path.exists(contact_pairs_path):
        df_pairs = pd.read_csv(contact_pairs_path)
        for _, row in df_pairs.iterrows():
            u_id, v_id = int(row['sender_id']), int(row['receiver_id'])
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])

    return adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids


# ==============================================================================
# FIGURE 3(d) MATRIX DIALOG
# ==============================================================================
class Fig3dMatrixDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure 3(d) Matrix - Actual vs GNN Predicted")
        self.setGeometry(100, 100, 750, 750)

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.fig.patch.set_facecolor('#111111')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        self.cmap_a = matplotlib.colormaps['YlOrRd']
        self.cmap_b = matplotlib.colormaps['YlGnBu']

    def draw_matrix(self, selected_nodes, weights_matrix_a, weights_matrix_b, active_min, active_max):
        self.ax.clear()
        self.ax.set_facecolor('#111111')

        N = len(selected_nodes)
        if N == 0:
            self.ax.text(0.5, 0.5, "No Particles Selected\n(Select particles on Left Window)", 
                         color='white', ha='center', va='center', fontsize=12)
            self.ax.axis('off')
            self.canvas.draw()
            return

        denom = (active_max - active_min) if active_max > active_min else 1.0
        font_sz = 7 if N <= 6 else (5 if N <= 12 else 3.5)

        pad_x = 0.08
        pad_bottom = 0.06
        usable_w = (1.0 - 2 * pad_x) / 2.0
        usable_h = 1.0 - 2 * pad_bottom

        for i in range(N):
            for j in range(N):
                u = selected_nodes[i]
                v = selected_nodes[j]

                w_a = weights_matrix_a.get((u, v), 0.0) if u != v else weights_matrix_a.get((u, u), 0.0)
                w_b = weights_matrix_b.get((u, v), 0.0) if u != v else weights_matrix_b.get((u, u), 0.0)

                x = j
                y = N - 1 - i

                is_diagonal = (i == j)
                bg_color = '#D8D8D8' if is_diagonal else '#FFFFFF'

                rect_bg = patches.Rectangle((x, y), 1.0, 1.0, 
                                            facecolor=bg_color, edgecolor='#B0B0B0', linewidth=0.6)
                self.ax.add_patch(rect_bg)

                if w_a > 0 or w_b > 0:
                    norm_a = np.clip((w_a - active_min) / denom, 0.0, 1.0) if w_a > 0 else 0.0
                    norm_b = np.clip((w_b - active_min) / denom, 0.0, 1.0) if w_b > 0 else 0.0

                    height_a = norm_a * usable_h
                    height_b = norm_b * usable_h

                    color_a = self.cmap_a(norm_a)
                    color_b = self.cmap_b(norm_b)

                    # Left Bar (Volume A - Actual)
                    if w_a > 0:
                        bar_a_x = x + pad_x
                        bar_a_y = y + pad_bottom
                        bar_a = patches.Rectangle((bar_a_x, bar_a_y), usable_w, height_a, 
                                                  facecolor=color_a, edgecolor='#222222', linewidth=0.3)
                        self.ax.add_patch(bar_a)

                        lum_a = 0.299 * color_a[0] + 0.587 * color_a[1] + 0.114 * color_a[2]
                        txt_color_a = 'black' if lum_a > 0.6 else 'white'
                        self.ax.text(bar_a_x + usable_w / 2.0, bar_a_y + max(height_a / 2.0, 0.12), f"{w_a:.2f}", 
                                     color=txt_color_a, fontsize=font_sz, fontweight='bold',
                                     ha='center', va='center', rotation=90)

                    # Right Bar (Volume B - Predicted)
                    if w_b > 0:
                        bar_b_x = x + pad_x + usable_w
                        bar_b_y = y + pad_bottom
                        bar_b = patches.Rectangle((bar_b_x, bar_b_y), usable_w, height_b, 
                                                  facecolor=color_b, edgecolor='#222222', linewidth=0.3)
                        self.ax.add_patch(bar_b)

                        lum_b = 0.299 * color_b[0] + 0.587 * color_b[1] + 0.114 * color_b[2]
                        txt_color_b = 'black' if lum_b > 0.6 else 'white'
                        self.ax.text(bar_b_x + usable_w / 2.0, bar_b_y + max(height_b / 2.0, 0.12), f"{w_b:.2f}", 
                                     color=txt_color_b, fontsize=font_sz, fontweight='bold',
                                     ha='center', va='center', rotation=90)

        self.ax.set_xlim(0, N)
        self.ax.set_ylim(0, N)

        self.ax.set_xticks([i + 0.5 for i in range(N)])
        self.ax.set_yticks([N - 1 - i + 0.5 for i in range(N)])
        self.ax.set_xticklabels([str(n) for n in selected_nodes], color='cyan', fontsize=9, rotation=90)
        self.ax.set_yticklabels([str(n) for n in selected_nodes], color='cyan', fontsize=9)

        self.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        self.ax.set_title("Fig 3(d) Matrix (Left Bar: Vol A [Heat] | Right Bar: Vol B [Cool])", color='white', pad=20, fontsize=11)

        self.fig.tight_layout()
        self.canvas.draw()


# ==============================================================================
# MAIN DUAL-PANEL VIEWER
# ==============================================================================
class DualPanelForceChainBuilder(QMainWindow):
    def __init__(self, adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids):
        super().__init__()
        self.setWindowTitle("Particle Selector & Force Chain Comparison (Assembly 58, Timestep 80)")
        self.setGeometry(50, 50, 1500, 850)

        self.adj_matrix = adj_matrix
        self.weights_a = weights_a
        self.weights_b = weights_b
        self.node_weights_a = node_weights_a
        self.node_weights_b = node_weights_b
        self.coords = coords
        self.ball_ids = ball_ids
        self.num_nodes = len(coords)

        all_w = np.concatenate([weights_a[weights_a > 0], weights_b[weights_b > 0]])
        self.global_min_w = float(np.min(all_w)) if len(all_w) > 0 else 0.1
        self.global_max_w = float(np.max(all_w)) if len(all_w) > 0 else 1.0
        self.active_min_w = self.global_min_w
        self.active_max_w = self.global_max_w

        self.selected_node_indices = set()
        self.hovered_node_idx = -1
        self.active_visible_nodes_left = []
        self.active_visible_nodes_right = []

        self.COLOR_NORMAL = (200, 200, 200)
        self.COLOR_HOVER = (0, 230, 255)
        self.COLOR_SELECTED = (255, 230, 0)

        self.build_graphs_from_arrays()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()

        initial_threshold_val = np.percentile(self.raw_weights, 65) if len(self.raw_weights) > 0 else 0.0
        self.slider_label = QLabel(f"Weight Threshold Value: {initial_threshold_val:.2f}")
        self.slider_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(65)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.on_slider_changed)

        self.lock_camera_checkbox = QCheckBox("Lock Camera Movement")
        self.lock_camera_checkbox.setChecked(True)
        self.lock_camera_checkbox.setStyleSheet("font-weight: bold; color: #00e6ff; font-size: 13px;")
        self.lock_camera_checkbox.toggled.connect(self.on_camera_lock_toggled)

        self.matrix_btn = QPushButton("Open Matrix View")
        self.matrix_btn.setStyleSheet("font-weight: bold; padding: 4px 12px; background-color: #005588; color: white;")
        self.matrix_btn.clicked.connect(self.open_matrix_window)

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.clear_btn.clicked.connect(self.clear_selection)

        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.threshold_slider)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.lock_camera_checkbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.matrix_btn)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.clear_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        self.setup_left_panel()
        self.setup_right_panel()

        self.enable_camera_sync(True)
        self.splitter.setSizes([700, 700])

        self.matrix_dialog = Fig3dMatrixDialog(self)

        self.update_left_visualization(threshold_percentile=65)
        self.ren_left.ResetCamera()

    def build_graphs_from_arrays(self):
        self.G_a = nx.Graph()
        self.G_b = nx.Graph()

        for i in range(self.num_nodes):
            self.G_a.add_node(i, pos=self.coords[i], weight=self.node_weights_a[i])
            self.G_b.add_node(i, pos=self.coords[i], weight=self.node_weights_b[i])

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i, j] == 1:
                    wa = self.weights_a[i, j]
                    wb = self.weights_b[i, j]
                    if wa > 0:
                        self.G_a.add_edge(i, j, weight=wa)
                    if wb > 0:
                        self.G_b.add_edge(i, j, weight=wb)

        # Maximum Spanning Tree backbone extraction
        self.mst_a = nx.maximum_spanning_tree(self.G_a, weight='weight')
        self.mst_b = nx.maximum_spanning_tree(self.G_b, weight='weight')

        all_edge_weights = []
        for _, _, d in self.mst_a.edges(data=True):
            all_edge_weights.append(d['weight'])
        for _, _, d in self.mst_b.edges(data=True):
            all_edge_weights.append(d['weight'])
        self.raw_weights = np.array(all_edge_weights, dtype=np.float32)

    def setup_left_panel(self):
        left_container = QFrame()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)

        self.vtk_left = QVTKRenderWindowInteractor(left_container)
        left_layout.addWidget(self.vtk_left)

        self.ren_left = vtk.vtkRenderer()
        self.ren_left.SetBackground(0.08, 0.08, 0.10)
        self.vtk_left.GetRenderWindow().AddRenderer(self.ren_left)
        self.vtk_left.Initialize()

        self.style_left = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_left.SetInteractorStyle(self.style_left)

        self.vtk_left.AddObserver("MouseMoveEvent", self.on_left_mouse_move)
        self.vtk_left.AddObserver("LeftButtonPressEvent", self.on_left_mouse_click)
        self.style_left.AddObserver("InteractionEvent", self.on_left_interaction)

        self.picker_left = vtk.vtkCellPicker()
        self.picker_left.SetTolerance(0.005)

        self.left_nodes_polydata = vtk.vtkPolyData()
        self.left_particle_colors = vtk.vtkUnsignedCharArray()
        self.left_particle_colors.SetNumberOfComponents(3)

        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(0.02)
        sphere_src.SetThetaResolution(10)
        sphere_src.SetPhiResolution(10)

        self.left_glyph_nodes = vtk.vtkGlyph3D()
        self.left_glyph_nodes.SetSourceConnection(sphere_src.GetOutputPort())
        self.left_glyph_nodes.SetInputData(self.left_nodes_polydata)
        self.left_glyph_nodes.SetColorModeToColorByScalar()
        self.left_glyph_nodes.SetScaleModeToDataScalingOff()
        self.left_glyph_nodes.GeneratePointIdsOn()

        mapper_nodes = vtk.vtkPolyDataMapper()
        mapper_nodes.SetInputConnection(self.left_glyph_nodes.GetOutputPort())
        mapper_nodes.SetScalarModeToUsePointData()

        self.left_actor_nodes = vtk.vtkActor()
        self.left_actor_nodes.SetMapper(mapper_nodes)
        self.ren_left.AddActor(self.left_actor_nodes)

        self.picker_left.AddPickList(self.left_actor_nodes)
        self.picker_left.PickFromListOn()

        self.left_edge_lut = vtk.vtkLookupTable()
        self.left_edge_lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            self.left_edge_lut.SetTableValue(i, t, 0.2 * (1.0 - t), 1.0 - t, 1.0)
        self.left_edge_lut.Build()

        self.left_chains_polydata = vtk.vtkPolyData()
        self.left_chains_mapper = vtk.vtkPolyDataMapper()
        self.left_chains_mapper.SetInputData(self.left_chains_polydata)
        self.left_chains_mapper.SetLookupTable(self.left_edge_lut)
        self.left_chains_mapper.SetScalarModeToUseCellData()

        self.left_actor_chains = vtk.vtkActor()
        self.left_actor_chains.SetMapper(self.left_chains_mapper)
        self.left_actor_chains.GetProperty().SetLineWidth(1.5)
        self.ren_left.AddActor(self.left_actor_chains)

        self.splitter.addWidget(left_container)

    def setup_right_panel(self):
        right_container = QFrame()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)

        self.vtk_right = QVTKRenderWindowInteractor(right_container)
        right_layout.addWidget(self.vtk_right)

        self.ren_right = vtk.vtkRenderer()
        self.ren_right.SetBackground(0.06, 0.06, 0.08)
        self.vtk_right.GetRenderWindow().AddRenderer(self.ren_right)
        self.vtk_right.Initialize()

        self.style_right = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_right.SetInteractorStyle(self.style_right)

        self.vtk_right.AddObserver("MouseMoveEvent", self.on_right_mouse_move)
        self.vtk_right.AddObserver("LeftButtonPressEvent", self.on_right_mouse_click)
        self.style_right.AddObserver("InteractionEvent", self.on_right_interaction)

        self.picker_right = vtk.vtkCellPicker()
        self.picker_right.SetTolerance(0.005)

        self.right_nodes_polydata = vtk.vtkPolyData()
        self.right_particle_colors = vtk.vtkUnsignedCharArray()
        self.right_particle_colors.SetNumberOfComponents(3)

        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(0.02)
        sphere_src.SetThetaResolution(10)
        sphere_src.SetPhiResolution(10)

        self.right_glyph_nodes = vtk.vtkGlyph3D()
        self.right_glyph_nodes.SetSourceConnection(sphere_src.GetOutputPort())
        self.right_glyph_nodes.SetInputData(self.right_nodes_polydata)
        self.right_glyph_nodes.SetColorModeToColorByScalar()
        self.right_glyph_nodes.SetScaleModeToDataScalingOff()
        self.right_glyph_nodes.GeneratePointIdsOn()

        mapper_nodes_right = vtk.vtkPolyDataMapper()
        mapper_nodes_right.SetInputConnection(self.right_glyph_nodes.GetOutputPort())
        mapper_nodes_right.SetScalarModeToUsePointData()

        self.right_actor_nodes = vtk.vtkActor()
        self.right_actor_nodes.SetMapper(mapper_nodes_right)
        self.ren_right.AddActor(self.right_actor_nodes)

        self.picker_right.AddPickList(self.right_actor_nodes)
        self.picker_right.PickFromListOn()

        self.right_chains_polydata = vtk.vtkPolyData()
        self.right_chains_mapper = vtk.vtkPolyDataMapper()
        self.right_chains_mapper.SetInputData(self.right_chains_polydata)
        self.right_chains_mapper.SetLookupTable(self.left_edge_lut)
        self.right_chains_mapper.SetScalarModeToUseCellData()

        self.right_actor_chains = vtk.vtkActor()
        self.right_actor_chains.SetMapper(self.right_chains_mapper)
        self.right_actor_chains.GetProperty().SetLineWidth(1.5)
        self.ren_right.AddActor(self.right_actor_chains)

        self.splitter.addWidget(right_container)

    def enable_camera_sync(self, enable=True):
        if enable:
            self.ren_right.SetActiveCamera(self.ren_left.GetActiveCamera())
        else:
            independent_cam = vtk.vtkCamera()
            independent_cam.DeepCopy(self.ren_left.GetActiveCamera())
            self.ren_right.SetActiveCamera(independent_cam)

    def on_camera_lock_toggled(self, checked):
        self.enable_camera_sync(checked)
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def on_left_interaction(self, obj, event):
        if self.lock_camera_checkbox.isChecked():
            self.vtk_right.GetRenderWindow().Render()

    def on_right_interaction(self, obj, event):
        if self.lock_camera_checkbox.isChecked():
            self.vtk_left.GetRenderWindow().Render()

    def open_matrix_window(self):
        self.update_matrix_view()
        self.matrix_dialog.show()

    def update_matrix_view(self):
        selected_list = sorted(list(self.selected_node_indices))

        weights_matrix_a = {}
        weights_matrix_b = {}

        # Fill edge weights: both volumes use the corresponding edge weight for every active edge
        for u, v, attr in self.pruned_mst_a.edges(data=True):
            weights_matrix_a[(u, v)] = attr['weight']
            weights_matrix_a[(v, u)] = attr['weight']

            # Populate Volume B's edge weight from the predicted edge matrix
            w_b = self.weights_b[u, v] if self.weights_b[u, v] > 0 else min(self.node_weights_b[u], self.node_weights_b[v])
            weights_matrix_b[(u, v)] = w_b
            weights_matrix_b[(v, u)] = w_b

        # Fill diagonal node weights
        for n in selected_list:
            weights_matrix_a[(n, n)] = self.node_weights_a[n]
            weights_matrix_b[(n, n)] = self.node_weights_b[n]

        self.matrix_dialog.draw_matrix(selected_list, weights_matrix_a, weights_matrix_b, 
                                        self.active_min_w, self.active_max_w)

    def update_left_visualization(self, threshold_percentile):
        min_weight_threshold = np.percentile(self.raw_weights, threshold_percentile) if len(self.raw_weights) > 0 else 0.0

        self.slider_label.setText(f"Weight Threshold Value: {min_weight_threshold:.2f}")

        # Filter MST A & B by current threshold
        self.pruned_mst_a = self.mst_a.copy()
        nodes_remove_a = [n for n, attr in self.pruned_mst_a.nodes(data=True) if attr['weight'] < min_weight_threshold]
        self.pruned_mst_a.remove_nodes_from(nodes_remove_a)

        self.pruned_mst_b = self.mst_b.copy()
        nodes_remove_b = [n for n, attr in self.pruned_mst_b.nodes(data=True) if attr['weight'] < min_weight_threshold]
        self.pruned_mst_b.remove_nodes_from(nodes_remove_b)

        # Particle Disappearance: Filter particles exceeding threshold weight
        self.active_visible_nodes_left = [i for i in range(self.num_nodes) if self.node_weights_a[i] >= min_weight_threshold]
        self.active_visible_nodes_right = [i for i in range(self.num_nodes) if self.node_weights_b[i] >= min_weight_threshold]

        if len(self.active_visible_nodes_left) == 0:
            self.left_actor_nodes.SetVisibility(False)
            self.left_actor_chains.SetVisibility(False)
            self.vtk_left.GetRenderWindow().Render()
            return

        active_coords = self.coords[self.active_visible_nodes_left]
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(active_coords, deep=True))
        self.left_nodes_polydata.SetPoints(pts)

        self.update_particle_colors()
        self.left_actor_nodes.SetVisibility(True)

        lines = vtk.vtkCellArray()
        line_pts = vtk.vtkPoints()
        edge_weight_list = []

        point_idx = 0
        for u, v, attr in self.pruned_mst_a.edges(data=True):
            p1, p2 = self.coords[u], self.coords[v]
            edge_weight_list.append(attr['weight'])

            line_pts.InsertNextPoint(p1)
            line_pts.InsertNextPoint(p2)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_idx)
            line.GetPointIds().SetId(1, point_idx + 1)
            lines.InsertNextCell(line)

            point_idx += 2

        edge_weights = np.array(edge_weight_list, dtype=np.float32)
        
        weights_a_active = [attr['weight'] for _, _, attr in self.pruned_mst_a.edges(data=True)]
        weights_b_active = [attr['weight'] for _, _, attr in self.pruned_mst_b.edges(data=True)]
        combined_active_weights = weights_a_active + weights_b_active

        if len(combined_active_weights) > 0:
            self.active_min_w = float(np.min(combined_active_weights))
            self.active_max_w = float(np.max(combined_active_weights))
            if self.active_min_w == self.active_max_w:
                self.active_max_w += 1e-5

            self.left_edge_lut.SetTableRange(self.active_min_w, self.active_max_w)
            self.left_edge_lut.Build()
            self.left_chains_mapper.SetScalarRange(self.active_min_w, self.active_max_w)
            self.right_chains_mapper.SetScalarRange(self.active_min_w, self.active_max_w)

        edge_weight_array = numpy_to_vtk(edge_weights, deep=True)
        edge_weight_array.SetName("EdgeWeights")

        self.left_chains_polydata.SetPoints(line_pts)
        self.left_chains_polydata.SetLines(lines)
        self.left_chains_polydata.GetCellData().SetScalars(edge_weight_array)

        self.left_actor_chains.SetVisibility(len(edge_weight_list) > 0)
        self.vtk_left.GetRenderWindow().Render()

        self.update_right_subgraph()

    def update_particle_colors(self):
        self.left_particle_colors.Reset()
        for node_id in self.active_visible_nodes_left:
            if node_id == self.hovered_node_idx:
                self.left_particle_colors.InsertNextTypedTuple(self.COLOR_HOVER)
            elif node_id in self.selected_node_indices:
                self.left_particle_colors.InsertNextTypedTuple(self.COLOR_SELECTED)
            else:
                self.left_particle_colors.InsertNextTypedTuple(self.COLOR_NORMAL)

        self.left_nodes_polydata.GetPointData().SetScalars(self.left_particle_colors)
        self.left_nodes_polydata.GetPointData().Modified()
        self.left_glyph_nodes.Update()

        if hasattr(self, 'current_selected_nodes_right') and len(self.current_selected_nodes_right) > 0:
            self.right_particle_colors.Reset()
            for node_id in self.current_selected_nodes_right:
                if node_id == self.hovered_node_idx:
                    self.right_particle_colors.InsertNextTypedTuple(self.COLOR_HOVER)
                else:
                    self.right_particle_colors.InsertNextTypedTuple(self.COLOR_SELECTED)

            self.right_nodes_polydata.GetPointData().SetScalars(self.right_particle_colors)
            self.right_nodes_polydata.GetPointData().Modified()
            self.right_glyph_nodes.Update()

    def update_right_subgraph(self):
        self.current_selected_nodes_right = sorted(list(self.selected_node_indices))

        if len(self.current_selected_nodes_right) == 0:
            self.right_actor_nodes.SetVisibility(False)
            self.right_actor_chains.SetVisibility(False)
            self.vtk_right.GetRenderWindow().Render()
            if self.matrix_dialog.isVisible():
                self.update_matrix_view()
            return

        selected_coords = self.coords[self.current_selected_nodes_right]
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(selected_coords, deep=True))
        self.right_nodes_polydata.SetPoints(pts)

        self.update_particle_colors()
        self.right_actor_nodes.SetVisibility(True)

        lines = vtk.vtkCellArray()
        line_pts = vtk.vtkPoints()
        edge_weight_list = []

        point_idx = 0
        for u, v, attr in self.pruned_mst_a.edges(data=True):
            if u in self.selected_node_indices and v in self.selected_node_indices:
                p1, p2 = self.coords[u], self.coords[v]
                edge_weight_list.append(attr['weight'])

                line_pts.InsertNextPoint(p1)
                line_pts.InsertNextPoint(p2)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, point_idx)
                line.GetPointIds().SetId(1, point_idx + 1)
                lines.InsertNextCell(line)

                point_idx += 2

        if len(edge_weight_list) > 0:
            edge_weights = np.array(edge_weight_list, dtype=np.float32)
            edge_weight_array = numpy_to_vtk(edge_weights, deep=True)

            self.right_chains_polydata.SetPoints(line_pts)
            self.right_chains_polydata.SetLines(lines)
            self.right_chains_polydata.GetCellData().SetScalars(edge_weight_array)
            self.right_actor_chains.SetVisibility(True)
        else:
            self.right_actor_chains.SetVisibility(False)

        self.vtk_right.GetRenderWindow().Render()

        if self.matrix_dialog.isVisible():
            self.update_matrix_view()

    def highlight_particle_center(self, point_idx):
        if self.hovered_node_idx == point_idx:
            return

        self.hovered_node_idx = point_idx
        self.update_particle_colors()

        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def clear_selection(self):
        self.selected_node_indices.clear()
        self.update_particle_colors()
        self.update_right_subgraph()
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def get_picked_particle_id_left(self):
        click_pos = self.vtk_left.GetEventPosition()
        self.picker_left.Pick(click_pos[0], click_pos[1], 0, self.ren_left)
        
        actor = self.picker_left.GetActor()
        if actor is not None and actor == self.left_actor_nodes:
            cell_id = self.picker_left.GetCellId()
            if cell_id >= 0:
                glyph_output = self.left_glyph_nodes.GetOutput()
                point_ids = glyph_output.GetPointData().GetArray("InputPointIds")
                if point_ids:
                    cell = glyph_output.GetCell(cell_id)
                    pt_idx = cell.GetPointId(0)
                    local_idx = int(point_ids.GetTuple1(pt_idx))
                    if 0 <= local_idx < len(self.active_visible_nodes_left):
                        return self.active_visible_nodes_left[local_idx]

        return -1

    def get_picked_particle_id_right(self):
        click_pos = self.vtk_right.GetEventPosition()
        self.picker_right.Pick(click_pos[0], click_pos[1], 0, self.ren_right)
        
        actor = self.picker_right.GetActor()
        if actor is not None and actor == self.right_actor_nodes:
            cell_id = self.picker_right.GetCellId()
            if cell_id >= 0:
                glyph_output = self.right_glyph_nodes.GetOutput()
                point_ids = glyph_output.GetPointData().GetArray("InputPointIds")
                if point_ids:
                    cell = glyph_output.GetCell(cell_id)
                    pt_idx = cell.GetPointId(0)
                    local_idx = int(point_ids.GetTuple1(pt_idx))
                    if 0 <= local_idx < len(self.current_selected_nodes_right):
                        return self.current_selected_nodes_right[local_idx]

        return -1

    def on_slider_changed(self, value):
        self.update_left_visualization(threshold_percentile=value)

    def on_left_mouse_move(self, obj, event):
        picked_id = self.get_picked_particle_id_left()
        self.highlight_particle_center(picked_id)
        self.style_left.OnMouseMove()

    def on_left_mouse_click(self, obj, event):
        picked_id = self.get_picked_particle_id_left()
        if picked_id != -1:
            if picked_id in self.selected_node_indices:
                self.selected_node_indices.remove(picked_id)
            else:
                self.selected_node_indices.add(picked_id)

            self.update_particle_colors()
            self.update_right_subgraph()
            self.vtk_left.GetRenderWindow().Render()

        self.style_left.OnLeftButtonDown()

    def on_right_mouse_move(self, obj, event):
        picked_id = self.get_picked_particle_id_right()
        self.highlight_particle_center(picked_id)
        self.style_right.OnMouseMove()

    def on_right_mouse_click(self, obj, event):
        picked_id = self.get_picked_particle_id_right()
        if picked_id != -1 and picked_id in self.selected_node_indices:
            self.selected_node_indices.remove(picked_id)
            self.update_particle_colors()
            self.update_right_subgraph()
            self.vtk_left.GetRenderWindow().Render()

        self.style_right.OnLeftButtonDown()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids = load_real_and_predicted_data(
        assembly_id=58, 
        timestep=80
    )

    app = QApplication(sys.argv)
    viewer = DualPanelForceChainBuilder(adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()