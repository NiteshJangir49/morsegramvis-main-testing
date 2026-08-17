import os
import sys
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

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


def load_vtp_nodes_and_weights(vtp_path):
    """Parses 3D coordinates and scalar weights from a VTP polydata file."""
    if not os.path.exists(vtp_path):
        print(f"[Error] File not found: {vtp_path}")
        return None, None

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    scalars = polydata.GetPointData().GetScalars()

    if points is None or scalars is None:
        print("[Error] Missing points or scalar array in VTP file.")
        return None, None

    coords = vtk_to_numpy(points.GetData()).astype(np.float32)
    weights = vtk_to_numpy(scalars).astype(np.float32)

    return coords, weights


class Fig3dMatrixDialog(QDialog):
    """
    Replication of Figure 3(d) from Alper et al. (CHI 2013):
    - Left half-cell bar = Volume A (Heat Colormap: YlOrRd)
    - Right half-cell bar = Volume B (Coolness Colormap: YlGnBu)
    - Bars fit left, bottom, and right edges perfectly, with variable top heights.
    - Global Min & Max calculated across both Volume A and B combined.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure 3(d) Replicated - Side-by-Side Bar Matrix Comparison")
        self.setGeometry(100, 100, 750, 750)

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.fig.patch.set_facecolor('#111111')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        # Matplotlib colormap accessors
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

        for i in range(N):
            for j in range(N):
                u = selected_nodes[i]
                v = selected_nodes[j]

                w_a = weights_matrix_a.get((u, v), 0.0) if u != v else weights_matrix_a.get((u, u), 0.0)
                w_b = weights_matrix_b.get((u, v), 0.0) if u != v else weights_matrix_b.get((u, u), 0.0)

                x = j
                y = N - 1 - i

                rect_bg = patches.Rectangle((x, y), 1.0, 1.0, 
                                            facecolor='#FFFFFF', edgecolor='#CCCCCC', linewidth=0.5)
                self.ax.add_patch(rect_bg)

                if w_a > 0 or w_b > 0:
                    norm_a = np.clip((w_a - active_min) / denom, 0.0, 1.0) if w_a > 0 else 0.0
                    norm_b = np.clip((w_b - active_min) / denom, 0.0, 1.0) if w_b > 0 else 0.0

                    height_a = norm_a
                    height_b = norm_b

                    color_a = self.cmap_a(norm_a)
                    color_b = self.cmap_b(norm_b)

                    if w_a > 0:
                        bar_a = patches.Rectangle((x, y), 0.50, height_a, 
                                                  facecolor=color_a, edgecolor='#222222', linewidth=0.3)
                        self.ax.add_patch(bar_a)

                        lum_a = 0.299 * color_a[0] + 0.587 * color_a[1] + 0.114 * color_a[2]
                        txt_color_a = 'black' if lum_a > 0.6 else 'white'

                        self.ax.text(x + 0.25, y + max(height_a / 2.0, 0.15), f"{w_a:.2f}", 
                                     color=txt_color_a, fontsize=font_sz, fontweight='bold',
                                     ha='center', va='center', rotation=90)

                    if w_b > 0:
                        bar_b = patches.Rectangle((x + 0.50, y), 0.50, height_b, 
                                                  facecolor=color_b, edgecolor='#222222', linewidth=0.3)
                        self.ax.add_patch(bar_b)

                        lum_b = 0.299 * color_b[0] + 0.587 * color_b[1] + 0.114 * color_b[2]
                        txt_color_b = 'black' if lum_b > 0.6 else 'white'

                        self.ax.text(x + 0.75, y + max(height_b / 2.0, 0.15), f"{w_b:.2f}", 
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


class DualPanelForceChainBuilder(QMainWindow):
    def __init__(self, coords_a, weights_a, coords_b, weights_b):
        super().__init__()
        self.setWindowTitle("Particle selector")
        self.setGeometry(50, 50, 1500, 850)

        self.coords_a = coords_a
        self.weights_a = weights_a
        self.coords_b = coords_b
        self.weights_b = weights_b

        self.coords = (coords_a + coords_b) / 2.0
        self.raw_weights = (weights_a + weights_b) / 2.0
        self.num_nodes = len(self.raw_weights)

        self.global_min_w = float(min(np.min(weights_a), np.min(weights_b)))
        self.global_max_w = float(max(np.max(weights_a), np.max(weights_b)))
        self.active_min_w = self.global_min_w
        self.active_max_w = self.global_max_w

        self.selected_node_indices = set()
        self.hovered_node_idx = -1
        self.active_visible_nodes_left = list(range(self.num_nodes))
        self.text_label_actors_right = []

        self.COLOR_NORMAL = (200, 200, 200)   # Light Gray
        self.COLOR_HOVER = (0, 230, 255)      # Cyan
        self.COLOR_SELECTED = (255, 230, 0)   # Yellow

        self.precompute_mst_graphs(k_neighbors=8)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()

        # Initial calculation for initial label text
        initial_threshold_val = np.percentile(self.raw_weights, 65)
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

        self.show_ids_checkbox = QCheckBox("Show Particle IDs")
        self.show_ids_checkbox.setChecked(True)
        self.show_ids_checkbox.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 13px;")
        self.show_ids_checkbox.toggled.connect(self.on_show_ids_toggled)

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
        control_layout.addWidget(self.show_ids_checkbox)
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

    def precompute_mst_graphs(self, k_neighbors=8):
        tree = KDTree(self.coords)

        # Graph A
        self.G_a = nx.Graph()
        for i in range(self.num_nodes):
            self.G_a.add_node(i, pos=self.coords[i], weight=self.weights_a[i])
        for i in range(self.num_nodes):
            distances, indices = tree.query(self.coords[i], k=k_neighbors + 1)
            for dist, j in zip(distances[1:], indices[1:]):
                # avg_w = (self.weights_a[i] + self.weights_a[j]) / 2.0
                avg_w = min(self.weights_a[i], self.weights_a[j])  # Optional: Use min instead of average
                self.G_a.add_edge(i, j, weight=avg_w)
        self.mst_a = nx.maximum_spanning_tree(self.G_a, weight='weight')

        # Graph B
        self.G_b = nx.Graph()
        for i in range(self.num_nodes):
            self.G_b.add_node(i, pos=self.coords[i], weight=self.weights_b[i])
        for i in range(self.num_nodes):
            distances, indices = tree.query(self.coords[i], k=k_neighbors + 1)
            for dist, j in zip(distances[1:], indices[1:]):
                # avg_w = (self.weights_b[i] + self.weights_b[j]) / 2.0
                avg_w = min(self.weights_b[i], self.weights_b[j])  # Optional: Use min instead of average
                self.G_b.add_edge(i, j, weight=avg_w)
        self.mst_b = nx.maximum_spanning_tree(self.G_b, weight='weight')

    def setup_left_panel(self):
        left_container = QFrame()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # header = QLabel("Interactive Selector & Full Network")
        # header.setStyleSheet("font-weight: bold; background-color: #222; color: #fff; padding: 4px;")
        # left_layout.addWidget(header)

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
        sphere_src.SetRadius(0.40)
        sphere_src.SetThetaResolution(12)
        sphere_src.SetPhiResolution(12)

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
        self.left_actor_chains.GetProperty().SetLineWidth(3.0)
        self.ren_left.AddActor(self.left_actor_chains)

        self.splitter.addWidget(left_container)

    def setup_right_panel(self):
        right_container = QFrame()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)

        # header = QLabel("Constructed Subgraph View (Click Particles to Deselect)")
        # header.setStyleSheet("font-weight: bold; background-color: #222; color: #fff; padding: 4px;")
        # right_layout.addWidget(header)

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
        sphere_src.SetRadius(0.40)
        sphere_src.SetThetaResolution(12)
        sphere_src.SetPhiResolution(12)

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
        self.right_actor_chains.GetProperty().SetLineWidth(3.5)
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

    def on_show_ids_toggled(self, checked):
        self.update_right_particle_id_labels()
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

        for u, v, attr in self.pruned_mst_a.edges(data=True):
            weights_matrix_a[(u, v)] = attr['weight']
            weights_matrix_a[(v, u)] = attr['weight']

        for u, v, attr in self.pruned_mst_b.edges(data=True):
            weights_matrix_b[(u, v)] = attr['weight']
            weights_matrix_b[(v, u)] = attr['weight']

        for n in selected_list:
            if n in self.pruned_mst_a.nodes():
                weights_matrix_a[(n, n)] = self.weights_a[n]
            if n in self.pruned_mst_b.nodes():
                weights_matrix_b[(n, n)] = self.weights_b[n]

        self.matrix_dialog.draw_matrix(selected_list, weights_matrix_a, weights_matrix_b, 
                                        self.active_min_w, self.active_max_w)

    def update_left_visualization(self, threshold_percentile):
        min_weight_threshold = np.percentile(self.raw_weights, threshold_percentile)
        
        # Display the actual scalar threshold value on GUI label
        self.slider_label.setText(f"Weight Threshold Value: {min_weight_threshold:.2f}")

        self.pruned_mst_a = self.mst_a.copy()
        nodes_remove_a = [n for n, attr in self.pruned_mst_a.nodes(data=True) if attr['weight'] < min_weight_threshold]
        self.pruned_mst_a.remove_nodes_from(nodes_remove_a)

        self.pruned_mst_b = self.mst_b.copy()
        nodes_remove_b = [n for n, attr in self.pruned_mst_b.nodes(data=True) if attr['weight'] < min_weight_threshold]
        self.pruned_mst_b.remove_nodes_from(nodes_remove_b)

        active_nodes = set(self.pruned_mst_a.nodes()).union(set(self.pruned_mst_b.nodes()))
        self.active_visible_nodes_left = sorted(list(active_nodes))

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

        self.left_actor_chains.SetVisibility(True)
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

    def update_right_particle_id_labels(self):
        for actor in self.text_label_actors_right:
            self.ren_right.RemoveActor(actor)
        self.text_label_actors_right.clear()

        if not self.show_ids_checkbox.isChecked():
            return

        active_cam = self.ren_right.GetActiveCamera()

        for node_id in self.current_selected_nodes_right:
            pos = self.coords[node_id]

            txt_source = vtk.vtkVectorText()
            txt_source.SetText(f"ID:{node_id}")

            txt_mapper = vtk.vtkPolyDataMapper()
            txt_mapper.SetInputConnection(txt_source.GetOutputPort())

            txt_actor = vtk.vtkFollower()
            txt_actor.SetMapper(txt_mapper)
            txt_actor.SetCamera(active_cam)
            txt_actor.SetScale(0.35, 0.35, 0.35)
            txt_actor.SetPosition(pos[0] + 0.45, pos[1] + 0.45, pos[2] + 0.45)
            txt_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            txt_actor.GetProperty().SetLighting(False)

            self.ren_right.AddActor(txt_actor)
            self.text_label_actors_right.append(txt_actor)

    def update_right_subgraph(self):
        self.current_selected_nodes_right = sorted(list(self.selected_node_indices))

        if len(self.current_selected_nodes_right) == 0:
            self.right_actor_nodes.SetVisibility(False)
            self.right_actor_chains.SetVisibility(False)
            self.update_right_particle_id_labels()
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

        self.update_right_particle_id_labels()

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

    base_dir = r"C:\Users\nites\Downloads\morsegramvis-main\morsegramvis-main"
    vtp_path_a = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")
    vtp_path_b = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_B.vtp")

    coords_a, weights_a = load_vtp_nodes_and_weights(vtp_path_a)
    coords_b, weights_b = load_vtp_nodes_and_weights(vtp_path_b)

    if coords_a is None or weights_a is None or coords_b is None or weights_b is None:
        return

    app = QApplication(sys.argv)
    viewer = DualPanelForceChainBuilder(coords_a, weights_a, coords_b, weights_b)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()