import os
import sys
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSlider, QComboBox, QCheckBox, QLineEdit, QPushButton
)
from PySide6.QtCore import Qt
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# ==============================================================================
# DATA LOADING UTILITY (Dynamic Assembly & Timestep)
# ==============================================================================
def load_real_and_predicted_data(assembly_id=58, timestep=80):
    """Loads 3D coordinates, actual NPMNCF (Vol A, normalized), and GNN predicted NPMNCF (Vol B)."""
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

    for path in [pos_x_path, pos_y_path, pos_z_path, actual_npmncf_path, pred_npmncf_path]:
        if not os.path.exists(path):
            print(f"[Error] Missing required file for Timestep {timestep}: {path}")
            return None, None, None

    # 1. Load 3D Coordinates
    df_x = pd.read_csv(pos_x_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'x'])
    df_y = pd.read_csv(pos_y_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'y'])
    df_z = pd.read_csv(pos_z_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'z'])

    df_x['ball_id'] = df_x['ball_id'].astype(int)
    df_y['ball_id'] = df_y['ball_id'].astype(int)
    df_z['ball_id'] = df_z['ball_id'].astype(int)

    df_coords = df_x.merge(df_y, on='ball_id').merge(df_z, on='ball_id')

    # 2. Load Actual NPMNCF (Volume A) and normalize
    df_actual = pd.read_csv(actual_npmncf_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'npmncf_raw'])
    df_actual['ball_id'] = pd.to_numeric(df_actual['ball_id'], errors='coerce')
    df_actual.dropna(subset=['ball_id'], inplace=True)
    df_actual['ball_id'] = df_actual['ball_id'].astype(int)

    raw_mean_force = df_actual['npmncf_raw'].mean()
    df_actual['node_weight_actual'] = df_actual['npmncf_raw'] / raw_mean_force if raw_mean_force > 0 else df_actual['npmncf_raw']

    normalized_mean_force = df_actual['node_weight_actual'].mean()
    print("\n" + "=" * 60)
    print(f"FORCE NORMALIZATION CHECK (Timestep {timestep} - Volume A):")
    print(f"  Raw Force Mean (raw NPMNCF)        : {raw_mean_force:.6f}")
    print(f"  Normalized Force Mean (Vol A Nodes): {normalized_mean_force:.6f}")
    print("=" * 60 + "\n")

    # 3. Load GNN Predicted NPMNCF (Volume B)
    df_pred = pd.read_csv(pred_npmncf_path)

    df_nodes = df_pred.merge(df_actual[['ball_id', 'node_weight_actual']], on='ball_id', how='inner')
    df_nodes = df_nodes.merge(df_coords, on='ball_id', how='inner')

    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float32) * 100.0
    weights_A = df_nodes['node_weight_actual'].values.astype(np.float32)
    weights_B = df_nodes['predicted_npmncf'].values.astype(np.float32)

    return coords, weights_A, weights_B


# ==============================================================================
# SPATIAL HEIGHTMAP & DIFFERENCE VIEWER
# ==============================================================================
class FullHeightmapViewer(QMainWindow):
    def __init__(self, initial_assembly=58, initial_timestep=80, base_dir=None):
        super().__init__()
        self.setWindowTitle(f"DiffSeer View - Spatial Heightmap (Assembly {initial_assembly}, Timestep {initial_timestep})")
        self.setGeometry(100, 100, 1750, 850)

        self.current_assembly = initial_assembly
        self.current_timestep = initial_timestep
        self.base_dir = base_dir

        self.volume_points = None
        self.volume_weights_A = None
        self.volume_weights_B = None

        coords, w_A, w_B = load_real_and_predicted_data(self.current_assembly, self.current_timestep)
        self.center_coords = coords
        self.weights_A = w_A
        self.weights_B = w_B

        self.compute_geometry_metrics()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()

        # Timestep Text Entry
        control_layout.addWidget(QLabel("<b>Timestep:</b>"))
        self.timestep_input = QLineEdit(str(self.current_timestep))
        self.timestep_input.setFixedWidth(50)
        self.timestep_input.setAlignment(Qt.AlignCenter)
        self.timestep_input.returnPressed.connect(self.handle_timestep_change)
        control_layout.addWidget(self.timestep_input)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(55)
        self.load_btn.setStyleSheet("font-weight: bold; background-color: #005588; color: white;")
        self.load_btn.clicked.connect(self.handle_timestep_change)
        control_layout.addWidget(self.load_btn)

        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("<b>Mode:</b>"))
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems([
            "Value Heightmaps",
            "A-B difference Heightmap"
        ])
        self.mode_dropdown.currentIndexChanged.connect(lambda idx: self.handle_mode_switch(idx, reset_cam=True))
        control_layout.addWidget(self.mode_dropdown)

        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("<b>Histogram Source:</b>"))
        self.hist_source_combo = QComboBox()
        self.hist_source_combo.addItems([
            "Centroids",
            "Volume"
        ])
        self.hist_source_combo.currentIndexChanged.connect(self.handle_hist_source_switch)
        control_layout.addWidget(self.hist_source_combo)

        control_layout.addSpacing(10)

        self.diff_toggle = QCheckBox("Volume Difference View")
        self.diff_toggle.setChecked(True)
        self.diff_toggle.stateChanged.connect(self.handle_diff_toggle_change)
        control_layout.addWidget(self.diff_toggle)

        control_layout.addSpacing(10)

        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.setChecked(False)
        self.bbox_toggle.stateChanged.connect(self.handle_bbox_toggle)
        control_layout.addWidget(self.bbox_toggle)

        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("<b>Axis:</b>"))
        self.axis_dropdown = QComboBox()
        self.axis_dropdown.addItems(["Z-Axis", "Y-Axis", "X-Axis"])
        self.axis_dropdown.currentIndexChanged.connect(self.handle_axis_switch)
        control_layout.addWidget(self.axis_dropdown)

        control_layout.addSpacing(10)

        self.height_label = QLabel("<b>Height: 20%</b>")
        control_layout.addWidget(self.height_label)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(0)
        self.height_slider.setMaximum(100)
        self.height_slider.setValue(20)
        self.height_slider.setFixedWidth(70)
        self.height_slider.valueChanged.connect(lambda: self.handle_geometry_update(reset_cam=False))
        control_layout.addWidget(self.height_slider)

        control_layout.addSpacing(10)

        self.box_width_label = QLabel("<b>Box: 15%</b>")
        control_layout.addWidget(self.box_width_label)
        self.box_width_slider = QSlider(Qt.Horizontal)
        self.box_width_slider.setMinimum(0)
        self.box_width_slider.setMaximum(100)
        self.box_width_slider.setValue(15)
        self.box_width_slider.setFixedWidth(70)
        self.box_width_slider.valueChanged.connect(self.handle_box_width_update)
        control_layout.addWidget(self.box_width_slider)

        control_layout.addSpacing(10)

        self.particle_size_label = QLabel("<b>Size: 50%</b>")
        control_layout.addWidget(self.particle_size_label)
        self.particle_size_slider = QSlider(Qt.Horizontal)
        self.particle_size_slider.setMinimum(0)
        self.particle_size_slider.setMaximum(100)
        self.particle_size_slider.setValue(50)
        self.particle_size_slider.setFixedWidth(70)
        self.particle_size_slider.valueChanged.connect(self.handle_particle_size_update)
        control_layout.addWidget(self.particle_size_slider)

        control_layout.addSpacing(10)

        # Opacity Contrast / Power Curve Slider: γ in [0.5, 4.0]
        self.contrast_label = QLabel("<b>Opacity Contrast: 1.0x</b>")
        control_layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)   # 10 -> γ = 1.0 (linear baseline)
        self.contrast_slider.setMaximum(40)   # 40 -> γ = 4.0 (steep suppression of low values)
        self.contrast_slider.setValue(10)
        self.contrast_slider.setFixedWidth(75)
        self.contrast_slider.valueChanged.connect(self.handle_contrast_change)
        control_layout.addWidget(self.contrast_slider)

        control_layout.addSpacing(10)

        self.opacity_label_A = QLabel("<b>Vol A: 1.00</b>")
        control_layout.addWidget(self.opacity_label_A)
        self.opacity_slider_A = QSlider(Qt.Horizontal)
        self.opacity_slider_A.setMinimum(0)
        self.opacity_slider_A.setMaximum(100)
        self.opacity_slider_A.setValue(100)
        self.opacity_slider_A.setFixedWidth(70)
        self.opacity_slider_A.valueChanged.connect(self.handle_opacity_A_change)
        control_layout.addWidget(self.opacity_slider_A)

        control_layout.addSpacing(10)

        self.opacity_label_B = QLabel("<b>Vol B: 1.00</b>")
        control_layout.addWidget(self.opacity_label_B)
        self.opacity_slider_B = QSlider(Qt.Horizontal)
        self.opacity_slider_B.setMinimum(0)
        self.opacity_slider_B.setMaximum(100)
        self.opacity_slider_B.setValue(100)
        self.opacity_slider_B.setFixedWidth(70)
        self.opacity_slider_B.valueChanged.connect(self.handle_opacity_B_change)
        control_layout.addWidget(self.opacity_slider_B)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.09, 0.1)

        render_win = self.vtkWidget.GetRenderWindow()
        render_win.SetAlphaBitPlanes(1)
        render_win.SetMultiSamples(0)
        self.renderer.SetUseDepthPeeling(1)
        self.renderer.SetOcclusionRatio(0.0)
        self.renderer.SetMaximumNumberOfPeels(4)

        # Lookup Tables for Heightmap Bars
        self.lut_diff = vtk.vtkLookupTable()
        self.lut_diff.SetNumberOfTableValues(2)
        self.lut_diff.SetTableValue(0, 0.9, 0.2, 0.2, 1.0)
        self.lut_diff.SetTableValue(1, 0.2, 0.5, 0.9, 1.0)
        self.lut_diff.Build()

        self.lut_A = vtk.vtkLookupTable()
        self.lut_A.SetNumberOfTableValues(1)
        self.lut_A.SetTableValue(0, 0.1, 0.6, 0.9, 1.0)
        self.lut_A.Build()

        self.lut_B = vtk.vtkLookupTable()
        self.lut_B.SetNumberOfTableValues(1)
        self.lut_B.SetTableValue(0, 0.9, 0.5, 0.1, 1.0)
        self.lut_B.Build()

        self.actor_A = vtk.vtkActor()
        self.actor_B = vtk.vtkActor()
        self.actor_diff = vtk.vtkActor()

        for actor in [self.actor_A, self.actor_B, self.actor_diff]:
            prop = actor.GetProperty()
            prop.SetInterpolationToFlat()
            prop.BackfaceCullingOn()

        self.renderer.AddActor(self.actor_A)
        self.renderer.AddActor(self.actor_B)
        self.renderer.AddActor(self.actor_diff)

        self.poly_nodes = vtk.vtkPolyData()
        self.glyph_filter = vtk.vtkGlyph3D()
        self.actor_nodes = vtk.vtkActor()
        self.renderer.AddActor(self.actor_nodes)

        # Bounding Box Wireframe Pipeline
        self.cube_source_bbox = vtk.vtkCubeSource()
        mapper_bbox = vtk.vtkPolyDataMapper()
        mapper_bbox.SetInputConnection(self.cube_source_bbox.GetOutputPort())

        self.actor_bbox = vtk.vtkActor()
        self.actor_bbox.SetMapper(mapper_bbox)
        self.actor_bbox.GetProperty().SetRepresentationToWireframe()
        self.actor_bbox.GetProperty().SetColor(0.85, 0.85, 0.85)
        self.actor_bbox.GetProperty().SetLineWidth(1.5)
        self.actor_bbox.SetVisibility(False)

        self.renderer.AddActor(self.actor_bbox)

        self.setup_difference_particle_pipeline()

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.mode_dropdown.setCurrentIndex(1)
        self.handle_geometry_update(reset_cam=True)

    def compute_geometry_metrics(self):
        self.min_coords = self.center_coords.min(axis=0)
        self.max_coords = self.center_coords.max(axis=0)

        from scipy.spatial import cKDTree
        tree = cKDTree(self.center_coords)
        distances, _ = tree.query(self.center_coords, k=2)
        self.particle_radius = (np.median(distances[:, 1]) if len(distances) > 0 else 2.0) * 0.5
        self.max_safe_particle_radius = self.particle_radius * 1.3

    def handle_timestep_change(self):
        text = self.timestep_input.text().strip()
        try:
            new_timestep = int(text)
        except ValueError:
            print(f"[Warning] Invalid timestep integer: '{text}'")
            return

        coords, w_A, w_B = load_real_and_predicted_data(self.current_assembly, new_timestep)
        if coords is None or len(w_A) == 0:
            print(f"[Warning] Failed to load data for Timestep {new_timestep}. Keeping previous state.")
            return

        self.current_timestep = new_timestep
        self.center_coords = coords
        self.weights_A = w_A
        self.weights_B = w_B
        self.volume_points = None
        self.volume_weights_A = None
        self.volume_weights_B = None

        self.compute_geometry_metrics()
        self.setWindowTitle(f"DiffSeer View - Spatial Heightmap (Assembly {self.current_assembly}, Timestep {self.current_timestep})")

        if self.hist_source_combo.currentIndex() == 1:
            self.generate_volumetric_sphere_sampling()

        if self.bbox_toggle.isChecked():
            min_b, max_b = self.get_full_ensemble_bounds()
            self.cube_source_bbox.SetBounds(
                min_b[0], max_b[0],
                min_b[1], max_b[1],
                min_b[2], max_b[2]
            )
            self.cube_source_bbox.Update()

        self.handle_geometry_update(reset_cam=False)

    def generate_volumetric_sphere_sampling(self):
        """Generates internal volumetric/surface sample points for each spherical particle."""
        num_samples_per_particle = 24
        indices = np.arange(0, num_samples_per_particle, dtype=float) + 0.5
        phi = np.arccos(1.0 - 2.0 * indices / num_samples_per_particle)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * indices

        r_offsets = np.c_[
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi)
        ] * (self.particle_radius * 0.85)

        r_offsets = np.vstack([[[0.0, 0.0, 0.0]], r_offsets])
        n_offsets = len(r_offsets)

        all_pts = (self.center_coords[:, None, :] + r_offsets[None, :, :]).reshape(-1, 3)

        self.volume_points = all_pts
        self.volume_weights_A = np.repeat(self.weights_A, n_offsets)
        self.volume_weights_B = np.repeat(self.weights_B, n_offsets)

    def get_full_ensemble_bounds(self):
        r = self.max_safe_particle_radius
        min_b = self.min_coords - r
        max_b = self.max_coords + r
        return min_b, max_b

    def handle_bbox_toggle(self, state):
        is_visible = self.bbox_toggle.isChecked()
        if is_visible:
            min_b, max_b = self.get_full_ensemble_bounds()
            self.cube_source_bbox.SetBounds(
                min_b[0], max_b[0],
                min_b[1], max_b[1],
                min_b[2], max_b[2]
            )
            self.cube_source_bbox.Update()

        self.actor_bbox.SetVisibility(is_visible)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def handle_hist_source_switch(self, index):
        if index == 1 and self.volume_points is None:
            self.generate_volumetric_sphere_sampling()

        self.handle_geometry_update(reset_cam=False)

    def get_active_dataset_source(self, is_diff=False, requested_weights=None):
        if self.hist_source_combo.currentIndex() == 1:
            if self.volume_points is None:
                self.generate_volumetric_sphere_sampling()

            coords = self.volume_points
            if is_diff:
                active_weights = self.volume_weights_A - self.volume_weights_B
            else:
                active_weights = self.volume_weights_A if np.array_equal(requested_weights, self.weights_A) else self.volume_weights_B
        else:
            coords = self.center_coords
            if is_diff:
                active_weights = self.weights_A - self.weights_B
            else:
                active_weights = requested_weights

        return coords, active_weights

    def generate_spatial_histogram_mesh(self, weights, axis_idx, lut, scale, is_diff=False):
        rem_axes = [i for i in [0, 1, 2] if i != axis_idx]
        u_ax, v_ax = rem_axes[0], rem_axes[1]

        coords, active_weights = self.get_active_dataset_source(is_diff=is_diff, requested_weights=weights)

        non_empty_mask = np.abs(active_weights) > 1e-6 if not is_diff else np.abs(active_weights) >= 1e-4

        valid_coords = coords[non_empty_mask]
        valid_weights = active_weights[non_empty_mask]

        if len(valid_coords) == 0:
            return vtk.vtkPolyDataMapper()

        min_u, max_u = self.min_coords[u_ax], self.max_coords[u_ax]
        min_v, max_v = self.min_coords[v_ax], self.max_coords[v_ax]

        box_width = self.get_physical_box_width(axis_idx)
        slider_val = self.box_width_slider.value()

        if slider_val == 100:
            grid_res_u, grid_res_v = 1, 1
            max_u = min_u + box_width
            max_v = min_v + box_width
        else:
            grid_res_u = max(2, int(np.ceil((max_u - min_u) / box_width)))
            grid_res_v = max(2, int(np.ceil((max_v - min_v) / box_width)))
            max_u = min_u + (grid_res_u * box_width)
            max_v = min_v + (grid_res_v * box_width)

        u_pts = valid_coords[:, u_ax]
        v_pts = valid_coords[:, v_ax]

        u_edges = np.linspace(min_u, max_u, grid_res_u + 1)
        v_edges = np.linspace(min_v, max_v, grid_res_v + 1)

        counts, _, _ = np.histogram2d(u_pts, v_pts, bins=[u_edges, v_edges])
        sums, _, _ = np.histogram2d(u_pts, v_pts, bins=[u_edges, v_edges], weights=valid_weights)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_grid = np.nan_to_num(sums / counts)

        valid_mask = counts > 0
        if is_diff:
            valid_mask = valid_mask & (np.abs(avg_grid) >= 1e-4)

        valid_iu, valid_iv = np.where(valid_mask)
        if len(valid_iu) == 0:
            return vtk.vtkPolyDataMapper()

        avg_values = avg_grid[valid_iu, valid_iv]

        global_min_w = valid_weights.min() if len(valid_weights) > 0 else 0.0
        global_max_w = valid_weights.max() if len(valid_weights) > 0 else 1.0
        max_abs = max(abs(global_min_w), abs(global_max_w)) or 1.0

        safe_clearance_plane = self.max_coords[axis_idx] + 2.0
        base_h_plane = (safe_clearance_plane + (scale * 2.0)) if is_diff else safe_clearance_plane

        u_centers = (u_edges[valid_iu] + u_edges[valid_iu + 1]) / 2.0
        v_centers = (v_edges[valid_iv] + v_edges[valid_iv + 1]) / 2.0

        if is_diff:
            bar_heights = np.maximum(0.2, (np.abs(avg_values) / max_abs) * scale * 10.0)
            val_scalars = np.where(avg_values >= 0, 0.5, -0.5)
            h_centers = np.where(avg_values >= 0, base_h_plane + (bar_heights / 2.0), base_h_plane - (bar_heights / 2.0))
        else:
            bar_heights = np.maximum(0.2, ((avg_values - global_min_w) / max(1e-5, (global_max_w - global_min_w))) * scale * 10.0)
            val_scalars = np.zeros_like(avg_values)
            h_centers = base_h_plane + (bar_heights / 2.0)

        append_filter = vtk.vtkAppendPolyData()
        size_u = box_width
        size_v = box_width

        for uc, vc, hc, bh, sc in zip(u_centers, v_centers, h_centers, bar_heights, val_scalars):
            poly_box = self.make_cube_geometry(uc, vc, hc, size_u, size_v, bh, sc, u_ax, v_ax, axis_idx)
            append_filter.AddInputData(poly_box)

        append_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(append_filter.GetOutput())
        mapper.ScalarVisibilityOn()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(-1.0 if is_diff else 0.0, 1.0)
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

        return mapper

    def setup_difference_particle_pipeline(self):
        self.glyph_filter.SetInputData(self.poly_nodes)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(14)
        sphere.SetPhiResolution(14)
        sphere.Update()

        self.glyph_filter.SetSourceConnection(sphere.GetOutputPort())
        self.glyph_filter.SetScaleModeToScaleByVector()
        self.glyph_filter.SetColorModeToColorByScalar()

        self.mapper_nodes = vtk.vtkPolyDataMapper()
        self.mapper_nodes.SetInputConnection(self.glyph_filter.GetOutputPort())
        self.mapper_nodes.SetScalarModeToUsePointData()
        self.mapper_nodes.SetColorModeToDirectScalars()

        self.actor_nodes.SetMapper(self.mapper_nodes)
        self.actor_nodes.GetProperty().LightingOn()
        self.actor_nodes.GetProperty().SetAmbient(0.3)
        self.actor_nodes.GetProperty().SetDiffuse(0.7)

        self.mapper_nodes.SetResolveCoincidentTopologyToPolygonOffset()
        self.mapper_nodes.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

        self.update_particle_pipeline_arrays()

    def update_particle_pipeline_arrays(self):
        """
        Dynamically applies a non-linear power-law contrast curve (gamma) to particle opacity.
        Increasing the slider suppresses low-magnitude particles (making them more transparent)
        while boosting high-magnitude particles (keeping them solid and prominent).
        """
        vtk_node_pts = vtk.vtkPoints()
        for pt in self.center_coords:
            vtk_node_pts.InsertNextPoint(pt[0], pt[1], pt[2])
        self.poly_nodes.SetPoints(vtk_node_pts)

        mode_idx = self.mode_dropdown.currentIndex()
        slider_val = self.particle_size_slider.value()
        size_pct = slider_val / 100.0
        current_max_scale = self.max_safe_particle_radius * (size_pct * 3.0)

        gamma = self.contrast_slider.value() / 10.0

        num_pts = len(self.center_coords)
        size_vectors = vtk.vtkFloatArray()
        size_vectors.SetNumberOfComponents(3)
        size_vectors.SetNumberOfTuples(num_pts)

        rgba_colors = vtk.vtkUnsignedCharArray()
        rgba_colors.SetNumberOfComponents(4)
        rgba_colors.SetNumberOfTuples(num_pts)

        diff = self.weights_A - self.weights_B
        max_diff = np.max(np.abs(diff)) or 1.0
        is_dev_mode = self.diff_toggle.isChecked()

        if mode_idx == 1 and is_dev_mode:
            for i, val in enumerate(diff):
                if slider_val == 0:
                    norm_size = 0.0
                else:
                    norm_size = (abs(val) / max_diff) * current_max_scale
                    norm_size = max(0.02, norm_size)

                size_vectors.SetTuple3(i, norm_size, norm_size, norm_size)

                mag_norm = np.clip(abs(val) / max_diff, 0.0, 1.0)
                alpha = int(255.0 * (mag_norm ** gamma))
                alpha = max(5, min(255, alpha))

                if val >= 0:
                    rgba_colors.SetTuple4(i, 230, 50, 50, alpha)
                else:
                    rgba_colors.SetTuple4(i, 50, 130, 230, alpha)
        else:
            w_max = max(np.max(self.weights_A), np.max(self.weights_B)) or 1.0
            for i in range(num_pts):
                if slider_val == 0:
                    norm_size = 0.0
                else:
                    norm_size = max(0.05, current_max_scale * 0.3)

                size_vectors.SetTuple3(i, norm_size, norm_size, norm_size)

                mag_norm = np.clip(max(self.weights_A[i], self.weights_B[i]) / w_max, 0.0, 1.0)
                alpha = int(255.0 * (mag_norm ** gamma))
                alpha = max(5, min(255, alpha))

                rgba_colors.SetTuple4(i, 240, 240, 240, alpha)

        self.poly_nodes.GetPointData().SetVectors(size_vectors)
        self.poly_nodes.GetPointData().SetScalars(rgba_colors)

        self.poly_nodes.Modified()
        self.glyph_filter.Update()

    def handle_contrast_change(self, value):
        gamma = value / 10.0
        self.contrast_label.setText(f"<b>Opacity Contrast: {gamma:.1f}x</b>")
        self.update_particle_pipeline_arrays()
        self.vtkWidget.GetRenderWindow().Render()

    def handle_diff_toggle_change(self, state):
        self.update_particle_pipeline_arrays()
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def handle_particle_size_update(self, value):
        self.particle_size_label.setText(f"<b>Size: {int(value)}%</b>")
        self.update_particle_pipeline_arrays()
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def get_physical_box_width(self, axis_idx):
        rem_axes = [i for i in [0, 1, 2] if i != axis_idx]
        span_u = self.max_coords[rem_axes[0]] - self.min_coords[rem_axes[0]]
        span_v = self.max_coords[rem_axes[1]] - self.min_coords[rem_axes[1]]
        min_lateral_span = min(span_u, span_v) or 1.0

        slider_val = self.box_width_slider.value()
        if slider_val == 100:
            return max(span_u, span_v) or 1.0

        slider_pct = slider_val / 100.0
        mapped_width = min_lateral_span * (0.01 + slider_pct * 0.34)
        return max(0.1, mapped_width)

    def get_mapped_height_scale(self):
        slider_pct = self.height_slider.value() / 100.0
        return 0.5 + (slider_pct * 39.5)

    def make_cube_geometry(self, uc, vc, hc, su, sv, sh, scalar_val, u_ax, v_ax, axis_idx):
        cube = vtk.vtkCubeSource()
        dims = [0.0, 0.0, 0.0]
        dims[u_ax] = su
        dims[v_ax] = sv
        dims[axis_idx] = sh
        cube.SetXLength(dims[0])
        cube.SetYLength(dims[1])
        cube.SetZLength(dims[2])

        pos = [0.0, 0.0, 0.0]
        pos[u_ax] = uc
        pos[v_ax] = vc
        pos[axis_idx] = hc
        cube.SetCenter(pos[0], pos[1], pos[2])
        cube.Update()

        poly = cube.GetOutput()
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfValues(poly.GetNumberOfPoints())
        for p in range(poly.GetNumberOfPoints()):
            scalars.SetValue(p, scalar_val)
        poly.GetPointData().SetScalars(scalars)
        return poly

    def reset_camera_to_full_volume(self):
        axis_index = self.axis_dropdown.currentIndex()
        total_min = list(self.min_coords)
        total_max = list(self.max_coords)

        active_actor = self.actor_diff if self.mode_dropdown.currentIndex() == 1 else self.actor_A

        if active_actor and active_actor.GetMapper() and active_actor.GetMapper().GetInput():
            bounds = active_actor.GetBounds()
            if bounds[0] <= bounds[1]:
                total_min[axis_index] = min(total_min[axis_index], bounds[axis_index * 2])
                total_max[axis_index] = max(total_max[axis_index], bounds[axis_index * 2 + 1])

        span = total_max[axis_index] - total_min[axis_index]
        total_min[axis_index] -= span * 0.1
        total_max[axis_index] += span * 0.1

        self.renderer.ResetCamera(
            total_min[0], total_max[0],
            total_min[1], total_max[1],
            total_min[2], total_max[2]
        )
        self.renderer.ResetCameraClippingRange()
        if hasattr(self, 'vtkWidget') and self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

    def handle_axis_switch(self, axis_index):
        self.handle_geometry_update(reset_cam=False)

    def handle_box_width_update(self, value):
        self.box_width_label.setText(f"<b>Box: {int(value)}%</b>")
        self.handle_geometry_update(reset_cam=False)

    def handle_geometry_update(self, reset_cam=False):
        self.height_label.setText(f"<b>Height: {int(self.height_slider.value())}%</b>")
        current_scale = self.get_mapped_height_scale()
        axis_index = self.axis_dropdown.currentIndex()

        mapper_A = self.generate_spatial_histogram_mesh(self.weights_A, axis_index, self.lut_A, current_scale, is_diff=False)
        self.actor_A.SetMapper(mapper_A)

        mapper_B = self.generate_spatial_histogram_mesh(self.weights_B, axis_index, self.lut_B, current_scale, is_diff=False)
        self.actor_B.SetMapper(mapper_B)

        diff_weights = self.weights_A - self.weights_B
        mapper_diff = self.generate_spatial_histogram_mesh(diff_weights, axis_index, self.lut_diff, current_scale, is_diff=True)
        self.actor_diff.SetMapper(mapper_diff)

        self.handle_mode_switch(self.mode_dropdown.currentIndex(), reset_cam=reset_cam)

    def handle_mode_switch(self, index, reset_cam=False):
        if index == 0:
            self.diff_toggle.setVisible(False)
            self.diff_toggle.setEnabled(False)

            self.actor_A.SetVisibility(True)
            self.actor_B.SetVisibility(True)
            self.actor_diff.SetVisibility(False)

            self.opacity_slider_B.setVisible(True)
            self.opacity_label_B.setVisible(True)
            self.opacity_label_A.setText(f"<b>Vol A: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            self.opacity_label_B.setText(f"<b>Vol B: {self.opacity_slider_B.value()/100.0:.2f}</b>")

            self.actor_A.GetProperty().SetOpacity(self.opacity_slider_A.value() / 100.0)
            self.actor_B.GetProperty().SetOpacity(self.opacity_slider_B.value() / 100.0)
        else:
            self.diff_toggle.setVisible(True)
            self.diff_toggle.setEnabled(True)

            self.actor_A.SetVisibility(False)
            self.actor_B.SetVisibility(False)
            self.actor_diff.SetVisibility(True)

            self.opacity_slider_B.setVisible(False)
            self.opacity_label_B.setVisible(False)
            self.opacity_label_A.setText(f"<b>Diff: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            self.actor_diff.GetProperty().SetOpacity(self.opacity_slider_A.value() / 100.0)

        self.update_particle_pipeline_arrays()

        if reset_cam:
            self.reset_camera_to_full_volume()
        else:
            self.renderer.ResetCameraClippingRange()
            if hasattr(self, 'vtkWidget') and self.vtkWidget.GetRenderWindow():
                self.vtkWidget.GetRenderWindow().Render()

    def handle_opacity_A_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_A.setText(f"<b>Vol A: {alpha:.2f}</b>")
            if self.actor_A: self.actor_A.GetProperty().SetOpacity(alpha)
        else:
            self.opacity_label_A.setText(f"<b>Diff: {alpha:.2f}</b>")
            if self.actor_diff: self.actor_diff.GetProperty().SetOpacity(alpha)
        self.vtkWidget.GetRenderWindow().Render()

    def handle_opacity_B_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_B.setText(f"<b>Vol B: {alpha:.2f}</b>")
            if self.actor_B: self.actor_B.GetProperty().SetOpacity(alpha)
        self.vtkWidget.GetRenderWindow().Render()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    # Initial start: Assembly 58, Timestep 80
    viewer = FullHeightmapViewer(initial_assembly=58, initial_timestep=80)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()