import os
import ast
import numpy as np
import pandas as pd
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSlider, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def load_dual_node_volume_data(particles_csv, node_vtp_path_A, node_vtp_path_B):
    if not os.path.exists(particles_csv) or not os.path.exists(node_vtp_path_A) or not os.path.exists(node_vtp_path_B):
        print(f"[Error] Missing path references:\nCSV: {particles_csv}\nVTP A: {node_vtp_path_A}\nVTP B: {node_vtp_path_B}")
        return None, None, None

    df_part = pd.read_csv(particles_csv)
    center_coords = []
    for _, row in df_part.iterrows():
        try:
            centroid = row['centroid']
            if isinstance(centroid, str):
                centroid = ast.literal_eval(centroid.strip())
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            if np.isnan(cx) or np.isnan(cy) or np.isnan(cz): continue
            center_coords.append([cx, cy, cz])
        except Exception:
            continue
    center_coords = np.array(center_coords)

    reader_A = vtk.vtkXMLPolyDataReader()
    reader_A.SetFileName(node_vtp_path_A)
    reader_A.Update()
    poly_A = reader_A.GetOutput()
    
    reader_B = vtk.vtkXMLPolyDataReader()
    reader_B.SetFileName(node_vtp_path_B)
    reader_B.Update()
    poly_B = reader_B.GetOutput()

    if not poly_A or not poly_B or not poly_A.GetPointData().GetScalars() or not poly_B.GetPointData().GetScalars():
        print("[Error] VTP structures missing valid node scalar arrays.")
        return None, None, None

    scalars_A = poly_A.GetPointData().GetScalars()
    scalars_B = poly_B.GetPointData().GetScalars()
    num_points = scalars_A.GetNumberOfValues()
    
    weights_A = np.zeros(num_points)
    weights_B = np.zeros(num_points)
    for i in range(num_points):
        weights_A[i] = scalars_A.GetValue(i)
        weights_B[i] = scalars_B.GetValue(i)

    return center_coords, weights_A, weights_B


class FullHeightmapViewer(QMainWindow):
    def __init__(self, center_coords, weights_A, weights_B, base_dir=None):
        super().__init__()
        self.setWindowTitle("DiffSeer View - Spatial Volume & Difference Viewer")
        self.setGeometry(100, 100, 1600, 850)

        self.center_coords = center_coords
        self.weights_A = weights_A
        self.weights_B = weights_B
        self.base_dir = base_dir
        
        self.bd_grains_dir = os.path.join(base_dir, "dataset_files", "Steel Beads", "bd_grains") if base_dir else None

        self.vtp_surface_points = None
        self.vtp_weights_A = None
        self.vtp_weights_B = None

        self.min_coords = center_coords.min(axis=0)
        self.max_coords = center_coords.max(axis=0)
        
        from scipy.spatial import cKDTree
        tree = cKDTree(self.center_coords)
        distances, _ = tree.query(self.center_coords, k=2)
        self.max_safe_particle_radius = (np.min(distances[:, 1]) if len(distances) > 0 else 2.0) * 0.65

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("<b>Mode:</b>"))
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems([
            "Value Heightmaps",
            "A-B difference Heightmap"
        ])
        self.mode_dropdown.currentIndexChanged.connect(self.handle_mode_switch)
        control_layout.addWidget(self.mode_dropdown)
        
        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("<b>Histogram Source:</b>"))
        self.hist_source_combo = QComboBox()
        self.hist_source_combo.addItems([
            "Centroids",
            "Volume"
        ])
        self.hist_source_combo.currentIndexChanged.connect(self.handle_hist_source_switch)
        control_layout.addWidget(self.hist_source_combo)

        control_layout.addSpacing(15)

        self.diff_toggle = QCheckBox("Volume Difference View")
        self.diff_toggle.setChecked(True)
        self.diff_toggle.stateChanged.connect(self.handle_diff_toggle_change)
        control_layout.addWidget(self.diff_toggle)

        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("<b>Projection Axis:</b>"))
        self.axis_dropdown = QComboBox()
        self.axis_dropdown.addItems(["Z-Axis", "Y-Axis", "X-Axis"])
        self.axis_dropdown.currentIndexChanged.connect(self.handle_axis_switch)
        control_layout.addWidget(self.axis_dropdown)
        
        control_layout.addSpacing(15)
        
        self.height_label = QLabel("<b>Height Scale: 20%</b>")
        control_layout.addWidget(self.height_label)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(0)   
        self.height_slider.setMaximum(100)  
        self.height_slider.setValue(20)   
        self.height_slider.setFixedWidth(80)
        self.height_slider.valueChanged.connect(self.handle_geometry_update)
        control_layout.addWidget(self.height_slider)

        control_layout.addSpacing(15)

        self.box_width_label = QLabel("<b>Box Width: 15%</b>")
        control_layout.addWidget(self.box_width_label)
        self.box_width_slider = QSlider(Qt.Horizontal)
        self.box_width_slider.setMinimum(0)   
        self.box_width_slider.setMaximum(100)  
        self.box_width_slider.setValue(15)   
        self.box_width_slider.setFixedWidth(80)
        self.box_width_slider.valueChanged.connect(self.handle_box_width_update)
        control_layout.addWidget(self.box_width_slider)

        control_layout.addSpacing(15)

        self.particle_size_label = QLabel("<b>Particle Size: 50%</b>")
        control_layout.addWidget(self.particle_size_label)
        self.particle_size_slider = QSlider(Qt.Horizontal)
        self.particle_size_slider.setMinimum(0)
        self.particle_size_slider.setMaximum(100)
        self.particle_size_slider.setValue(50)
        self.particle_size_slider.setFixedWidth(80)
        self.particle_size_slider.valueChanged.connect(self.handle_particle_size_update)
        control_layout.addWidget(self.particle_size_slider)

        control_layout.addSpacing(15)
        
        self.opacity_label_A = QLabel("<b>Vol A Opacity: 1.00</b>")
        control_layout.addWidget(self.opacity_label_A)
        self.opacity_slider_A = QSlider(Qt.Horizontal)
        self.opacity_slider_A.setMinimum(0)
        self.opacity_slider_A.setMaximum(100)
        self.opacity_slider_A.setValue(100)
        self.opacity_slider_A.setFixedWidth(80)
        self.opacity_slider_A.valueChanged.connect(self.handle_opacity_A_change)
        control_layout.addWidget(self.opacity_slider_A)

        control_layout.addSpacing(15)

        self.opacity_label_B = QLabel("<b>Vol B Opacity: 1.00</b>")
        control_layout.addWidget(self.opacity_label_B)
        self.opacity_slider_B = QSlider(Qt.Horizontal)
        self.opacity_slider_B.setMinimum(0)
        self.opacity_slider_B.setMaximum(100)
        self.opacity_slider_B.setValue(100)
        self.opacity_slider_B.setFixedWidth(80)
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

        self.lut_diff = vtk.vtkLookupTable()
        self.lut_diff.SetNumberOfTableValues(2)
        self.lut_diff.SetTableValue(0, 0.9, 0.2, 0.2, 1.0) 
        self.lut_diff.SetTableValue(1, 0.2, 0.5, 0.9, 1.0) 
        self.lut_diff.Build()

        self.lut_A = vtk.vtkLookupTable()
        self.lut_A.SetNumberOfTableValues(1)
        self.lut_A.SetTableValue(0, 0.9, 0.6, 0.1, 1.0)
        self.lut_A.Build()

        self.lut_B = vtk.vtkLookupTable()
        self.lut_B.SetNumberOfTableValues(1)
        self.lut_B.SetTableValue(0, 0.1, 0.6, 0.9, 1.0)
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

        self.setup_difference_particle_pipeline()

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.mode_dropdown.setCurrentIndex(1)
        self.handle_axis_switch(0)

    def load_full_grain_ensemble_backend(self):
        """Loads raw grain points in the backend to drive fine-grained heightmaps."""
        if not self.bd_grains_dir or not os.path.exists(self.bd_grains_dir):
            print(f"[Error] Grain directory missing: {self.bd_grains_dir}")
            return

        vtp_files = sorted([
            os.path.join(self.bd_grains_dir, f) 
            for f in os.listdir(self.bd_grains_dir) 
            if f.endswith(".vtp")
        ])

        num_particles = min(len(self.center_coords), len(self.weights_A), len(vtp_files))
        if num_particles == 0:
            return

        print(f"[Backend Data Sync] Loading backend point-cloud arrays across {num_particles} grains...")

        all_pts = []
        wA_list = []
        wB_list = []

        for i in range(num_particles):
            wA = self.weights_A[i]
            wB = self.weights_B[i]

            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp_files[i])
            reader.Update()
            poly = reader.GetOutput()

            if not poly or poly.GetNumberOfPoints() == 0:
                continue

            n_pts = poly.GetNumberOfPoints()
            for p in range(n_pts):
                all_pts.append(poly.GetPoint(p))
                wA_list.append(wA)
                wB_list.append(wB)

        self.vtp_surface_points = np.array(all_pts)
        self.vtp_weights_A = np.array(wA_list)
        self.vtp_weights_B = np.array(wB_list)

        print(f"[Backend Data Sync] Loaded {len(all_pts)} dense sampling points for full-ensemble heightmaps.")

    def handle_hist_source_switch(self, index):
        if index == 1 and self.vtp_surface_points is None:
            self.load_full_grain_ensemble_backend()

        self.handle_geometry_update()

    def get_active_dataset_source(self, is_diff=False, requested_weights=None):
        if self.hist_source_combo.currentIndex() == 1 and self.vtp_surface_points is not None:
            coords = self.vtp_surface_points
            if is_diff:
                active_weights = self.vtp_weights_A - self.vtp_weights_B
            else:
                active_weights = self.vtp_weights_A if np.array_equal(requested_weights, self.weights_A) else self.vtp_weights_B
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

        bin_contents = [[[] for _ in range(grid_res_v)] for _ in range(grid_res_u)]
        for pt, w in zip(coords, active_weights):
            if slider_val == 100:
                iu, iv = 0, 0
            else:
                if pt[u_ax] < min_u or pt[u_ax] >= max_u or pt[v_ax] < min_v or pt[v_ax] >= max_v:
                    continue
                iu = int((pt[u_ax] - min_u) / box_width)
                iv = int((pt[v_ax] - min_v) / box_width)
                iu = min(iu, grid_res_u - 1)
                iv = min(iv, grid_res_v - 1)
            
            if is_diff and abs(w) < 1e-4:
                continue
            bin_contents[iu][iv].append(w)

        append_filter = vtk.vtkAppendPolyData()
        has_geometry = False

        global_min_w = active_weights.min() if len(active_weights) > 0 else 0.0
        global_max_w = active_weights.max() if len(active_weights) > 0 else 1.0
        max_abs = max(abs(global_min_w), abs(global_max_w)) or 1.0

        safe_clearance_plane = self.max_coords[axis_idx] + 2.0
        
        if is_diff:
            max_possible_height = scale * 10.0
            base_h_plane = safe_clearance_plane + max_possible_height
        else:
            base_h_plane = safe_clearance_plane

        bar_thickness_ratio = 1.0
        size_u = box_width * bar_thickness_ratio
        size_v = box_width * bar_thickness_ratio

        for iu in range(grid_res_u):
            u_center = (self.min_coords[u_ax] + self.max_coords[u_ax]) / 2.0 if slider_val == 100 else min_u + (iu + 0.5) * box_width
                
            for iv in range(grid_res_v):
                v_center = (self.min_coords[v_ax] + self.max_coords[v_ax]) / 2.0 if slider_val == 100 else min_v + (iv + 0.5) * box_width
                    
                cell_weights = np.array(bin_contents[iu][iv])
                if len(cell_weights) == 0: continue

                avg_value = np.mean(cell_weights)
                if abs(avg_value) < 1e-4: continue

                bar_height = abs(avg_value / max_abs) * scale * 10.0 if is_diff else ((avg_value - global_min_w) / max(1e-5, (global_max_w - global_min_w))) * scale * 10.0
                bar_height = max(0.2, bar_height) 

                if is_diff:
                    val_scalar = -0.5 if avg_value < 0 else 0.5
                    if avg_value < 0:
                        h_center = base_h_plane - (bar_height / 2.0)
                    else:
                        h_center = base_h_plane + (bar_height / 2.0)
                else:
                    val_scalar = 0.0
                    h_center = base_h_plane + (bar_height / 2.0)

                has_geometry = True
                poly_box = self.make_cube_geometry(u_center, v_center, h_center, size_u, size_v, bar_height, val_scalar, u_ax, v_ax, axis_idx)
                append_filter.AddInputData(poly_box)

        poly_data = vtk.vtkPolyData()
        if has_geometry:
            append_filter.Update()
            poly_data.ShallowCopy(append_filter.GetOutput())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOn()
        mapper.SetLookupTable(lut)
        
        if is_diff:
            mapper.SetScalarRange(-1.0, 1.0)
        else:
            mapper.SetScalarRange(0.0, 1.0)
        
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
        self.mapper_nodes.SetLookupTable(self.lut_diff)
        self.mapper_nodes.SetScalarRange(-1.0, 1.0)

        self.actor_nodes.SetMapper(self.mapper_nodes)
        self.actor_nodes.GetProperty().LightingOn()
        self.actor_nodes.GetProperty().SetAmbient(0.3)
        self.actor_nodes.GetProperty().SetDiffuse(0.7)
        
        self.mapper_nodes.SetResolveCoincidentTopologyToPolygonOffset()
        self.mapper_nodes.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

        self.update_particle_pipeline_arrays()

    def update_particle_pipeline_arrays(self):
        """Updates particle sphere colors and radii with clean zero-scale collapsing."""
        vtk_node_pts = vtk.vtkPoints()
        for pt in self.center_coords:
            vtk_node_pts.InsertNextPoint(pt[0], pt[1], pt[2])
        self.poly_nodes.SetPoints(vtk_node_pts)

        mode_idx = self.mode_dropdown.currentIndex()
        slider_val = self.particle_size_slider.value()
        size_pct = slider_val / 100.0
        current_max_scale = self.max_safe_particle_radius * (size_pct * 3.0)

        size_vectors = vtk.vtkFloatArray()
        size_vectors.SetNumberOfComponents(3)
        size_vectors.SetNumberOfTuples(len(self.center_coords))

        color_scalars = vtk.vtkFloatArray()
        color_scalars.SetNumberOfValues(len(self.center_coords))

        diff = self.weights_A - self.weights_B
        max_diff = np.max(np.abs(diff)) or 1.0
        is_dev_mode = self.diff_toggle.isChecked()

        if mode_idx == 1 and is_dev_mode:
            self.mapper_nodes.ScalarVisibilityOn()
            for i, val in enumerate(diff):
                if slider_val == 0:
                    norm_size = 0.0  
                else:
                    norm_size = (abs(val) / max_diff) * current_max_scale
                    norm_size = max(0.02, norm_size)  
                
                size_vectors.SetTuple3(i, norm_size, norm_size, norm_size)
                color_scalars.SetValue(i, -0.5 if val >= 0 else 0.5)
        else:
            self.mapper_nodes.ScalarVisibilityOff()
            self.actor_nodes.GetProperty().SetColor(1.0, 1.0, 1.0)
            
            for i in range(len(self.center_coords)):
                if slider_val == 0:
                    norm_size = 0.0  
                else:
                    norm_size = max(0.05, current_max_scale * 0.3)
                
                size_vectors.SetTuple3(i, norm_size, norm_size, norm_size)
                color_scalars.SetValue(i, 0.0)

        self.poly_nodes.GetPointData().SetVectors(size_vectors)
        self.poly_nodes.GetPointData().SetScalars(color_scalars)
        
        self.poly_nodes.Modified()
        self.glyph_filter.Update()

    def handle_diff_toggle_change(self, state):
        self.update_particle_pipeline_arrays()
        if self.mode_dropdown.currentIndex() == 1:
            if self.diff_toggle.isChecked():
                self.mapper_nodes.ScalarVisibilityOn()
            else:
                self.mapper_nodes.ScalarVisibilityOff()
                self.actor_nodes.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.vtkWidget.GetRenderWindow().Render()

    def handle_particle_size_update(self, value):
        self.particle_size_label.setText(f"<b>Particle Size: {int(value)}%</b>")
        self.update_particle_pipeline_arrays()
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
        self.handle_geometry_update()

    def handle_box_width_update(self, value):
        self.box_width_label.setText(f"<b>Box Width: {int(value)}%</b>")
        self.handle_geometry_update()

    def handle_geometry_update(self):
        self.height_label.setText(f"<b>Height Scale: {int(self.height_slider.value())}%</b>")
        current_scale = self.get_mapped_height_scale()
        axis_index = self.axis_dropdown.currentIndex()

        mapper_A = self.generate_spatial_histogram_mesh(self.weights_A, axis_index, self.lut_A, current_scale, is_diff=False)
        self.actor_A.SetMapper(mapper_A)

        mapper_B = self.generate_spatial_histogram_mesh(self.weights_B, axis_index, self.lut_B, current_scale, is_diff=False)
        self.actor_B.SetMapper(mapper_B)

        diff_weights = self.weights_A - self.weights_B
        mapper_diff = self.generate_spatial_histogram_mesh(diff_weights, axis_index, self.lut_diff, current_scale, is_diff=True)
        self.actor_diff.SetMapper(mapper_diff)

        self.handle_mode_switch(self.mode_dropdown.currentIndex())

    def handle_mode_switch(self, index):
        if index == 0:
            self.diff_toggle.setVisible(False)
            self.diff_toggle.setEnabled(False)

            self.actor_A.SetVisibility(True)
            self.actor_B.SetVisibility(True)
            self.actor_diff.SetVisibility(False)
            
            self.opacity_slider_B.setVisible(True)
            self.opacity_label_B.setVisible(True)
            self.opacity_label_A.setText(f"<b>Vol A Opacity: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            self.opacity_label_B.setText(f"<b>Vol B Opacity: {self.opacity_slider_B.value()/100.0:.2f}</b>")
            
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
            self.opacity_label_A.setText(f"<b>Diff Opacity: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            self.actor_diff.GetProperty().SetOpacity(self.opacity_slider_A.value() / 100.0)

        self.update_particle_pipeline_arrays()
        self.reset_camera_to_full_volume()

    def handle_opacity_A_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_A.setText(f"<b>Vol A Opacity: {alpha:.2f}</b>")
            if self.actor_A: self.actor_A.GetProperty().SetOpacity(alpha)
        else:
            self.opacity_label_A.setText(f"<b>Diff Opacity: {alpha:.2f}</b>")
            if self.actor_diff: self.actor_diff.GetProperty().SetOpacity(alpha)
        self.vtkWidget.GetRenderWindow().Render()

    def handle_opacity_B_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_B.setText(f"<b>Vol B Opacity: {alpha:.2f}</b>")
            if self.actor_B: self.actor_B.GetProperty().SetOpacity(alpha)
        self.vtkWidget.GetRenderWindow().Render()

def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    particles_file = os.path.join(base_dir, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
    node_vtp_A = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")
    node_vtp_B = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_B.vtp")

    center_coords, w_A, w_B = load_dual_node_volume_data(particles_file, node_vtp_A, node_vtp_B)
    
    if center_coords is None or len(w_A) == 0:
        return

    app = QApplication(sys.argv)
    viewer = FullHeightmapViewer(center_coords, w_A, w_B, base_dir=base_dir)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()