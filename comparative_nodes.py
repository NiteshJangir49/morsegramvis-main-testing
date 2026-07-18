import os
import ast
import numpy as np
import pandas as pd
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox
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
    def __init__(self, center_coords, weights_A, weights_B):
        super().__init__()
        self.setWindowTitle("DiffSeer view")
        self.setGeometry(100, 100, 1500, 850)

        self.center_coords = center_coords
        self.weights_A = weights_A
        self.weights_B = weights_B

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
        
        control_layout.addSpacing(20)

        control_layout.addWidget(QLabel("<b>Projection Axis:</b>"))
        self.axis_dropdown = QComboBox()
        self.axis_dropdown.addItems(["Z-Axis", "Y-Axis", "X-Axis"])
        self.axis_dropdown.currentIndexChanged.connect(self.handle_axis_switch)
        control_layout.addWidget(self.axis_dropdown)
        
        control_layout.addSpacing(20)
        
        self.height_label = QLabel("<b>Height Scale: 4.0x</b>")
        control_layout.addWidget(self.height_label)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(5)   
        self.height_slider.setMaximum(800)  
        self.height_slider.setValue(40)   
        self.height_slider.setFixedWidth(110)
        self.height_slider.valueChanged.connect(self.handle_geometry_update)
        control_layout.addWidget(self.height_slider)

        control_layout.addSpacing(20)
        
        self.opacity_label_A = QLabel("<b>Volume A Opacity: 1.00</b>")
        control_layout.addWidget(self.opacity_label_A)
        self.opacity_slider_A = QSlider(Qt.Horizontal)
        self.opacity_slider_A.setMinimum(0)
        self.opacity_slider_A.setMaximum(100)
        self.opacity_slider_A.setValue(100)
        self.opacity_slider_A.setFixedWidth(110)
        self.opacity_slider_A.valueChanged.connect(self.handle_opacity_A_change)
        control_layout.addWidget(self.opacity_slider_A)
        
        control_layout.addSpacing(15)
        
        self.opacity_label_B = QLabel("<b>Volume B Opacity: 1.00</b>")
        control_layout.addWidget(self.opacity_label_B)
        self.opacity_slider_B = QSlider(Qt.Horizontal)
        self.opacity_slider_B.setMinimum(0)
        self.opacity_slider_B.setMaximum(100)
        self.opacity_slider_B.setValue(100)
        self.opacity_slider_B.setFixedWidth(110)
        self.opacity_slider_B.valueChanged.connect(self.handle_opacity_B_change)
        control_layout.addWidget(self.opacity_slider_B)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.09, 0.1)

        self.grid_res_u = 50
        self.grid_res_v = 50
        self.min_coords = center_coords.min(axis=0)
        self.max_coords = center_coords.max(axis=0)

        self.lut_A = vtk.vtkLookupTable()
        self.lut_A.SetNumberOfTableValues(256)
        self.lut_A.SetHueRange(0.15, 0.0) 
        self.lut_A.Build()

        self.lut_B = vtk.vtkLookupTable()
        self.lut_B.SetNumberOfTableValues(256)
        self.lut_B.SetHueRange(0.55, 0.65)  
        self.lut_B.Build()

        self.lut_diff = vtk.vtkLookupTable()
        self.lut_diff.SetNumberOfTableValues(256)
        self.lut_diff.Build()
        for i in range(256):
            t = i / 255.0
            if t < 0.5:
                f = t * 2.0
                self.lut_diff.SetTableValue(i, 0.9, 0.2 * f, 0.2 * f, 1.0)
            else:
                f = (t - 0.5) * 2.0
                self.lut_diff.SetTableValue(i, 0.4 * (1.0 - f), 0.45 + (0.05 * f), 0.5 + (0.4 * f), 1.0)

        self.surface_points_A = vtk.vtkPoints()
        self.poly_A = vtk.vtkPolyData()
        self.actor_A = vtk.vtkActor()

        self.surface_points_B = vtk.vtkPoints()
        self.poly_B = vtk.vtkPolyData()
        self.actor_B = vtk.vtkActor()

        self.surface_points_diff = vtk.vtkPoints()
        self.poly_diff = vtk.vtkPolyData()
        self.actor_diff = vtk.vtkActor()

        self.renderer.AddActor(self.actor_A)
        self.renderer.AddActor(self.actor_B)
        self.renderer.AddActor(self.actor_diff)

        vtk_node_pts = vtk.vtkPoints()
        for pt in center_coords:
            vtk_node_pts.InsertNextPoint(pt[0], pt[1], pt[2])
        poly_nodes = vtk.vtkPolyData()
        poly_nodes.SetPoints(vtk_node_pts)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly_nodes)
        glyph.Update()
        mapper_nodes = vtk.vtkPolyDataMapper()
        mapper_nodes.SetInputConnection(glyph.GetOutputPort())
        actor_nodes = vtk.vtkActor()
        actor_nodes.SetMapper(mapper_nodes)
        actor_nodes.GetProperty().SetColor(0.5, 0.55, 0.6)
        actor_nodes.GetProperty().SetPointSize(3.0)
        actor_nodes.GetProperty().LightingOff()
        self.renderer.AddActor(actor_nodes)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.handle_axis_switch(0)
        self.reset_camera_to_full_volume()

    def compute_axis_averages(self, weights, axis_idx):
        rem_axes = [i for i in [0, 1, 2] if i != axis_idx]
        u_ax, v_ax = rem_axes[0], rem_axes[1]

        min_u, max_u = self.min_coords[u_ax], self.max_coords[u_ax]
        min_v, max_v = self.min_coords[v_ax], self.max_coords[v_ax]

        du = (max_u - min_u) / (self.grid_res_u - 1) if max_u != min_u else 1.0
        dv = (max_v - min_v) / (self.grid_res_v - 1) if max_v != min_v else 1.0

        accum_w = np.zeros((self.grid_res_u, self.grid_res_v))
        accum_c = np.zeros((self.grid_res_u, self.grid_res_v))

        for pt, w in zip(self.center_coords, weights):
            iu = min(int((pt[u_ax] - min_u) / du), self.grid_res_u - 1)
            iv = min(int((pt[v_ax] - min_v) / dv), self.grid_res_v - 1)
            accum_w[iu, iv] += w
            accum_c[iu, iv] += 1

        avg_grid = np.zeros((self.grid_res_u, self.grid_res_v))
        for iu in range(self.grid_res_u):
            for iv in range(self.grid_res_v):
                if accum_c[iu, iv] > 0:
                    avg_grid[iu, iv] = accum_w[iu, iv] / accum_c[iu, iv]
        return avg_grid

    def update_heightmap_geometry(self, normalized_grid, out_points, poly, actor, lut, scale, axis_idx, is_diff=False):
        rem_axes = [i for i in [0, 1, 2] if i != axis_idx]
        u_ax, v_ax = rem_axes[0], rem_axes[1]

        grid_u = np.linspace(self.min_coords[u_ax], self.max_coords[u_ax], self.grid_res_u)
        grid_v = np.linspace(self.min_coords[v_ax], self.max_coords[v_ax], self.grid_res_v)

        base_h_plane = self.max_coords[axis_idx] + 1.0
        min_val = normalized_grid.min()
        h_shift = abs(min_val) if (is_diff and min_val < 0) else 0.0
        adjusted_base_h = base_h_plane + (h_shift * scale)

        out_points.Reset()
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfValues(self.grid_res_u * self.grid_res_v)

        pt_coord = [0.0, 0.0, 0.0]
        idx = 0
        for iu in range(self.grid_res_u):
            for iv in range(self.grid_res_v):
                val = normalized_grid[iu, iv]
                h_val = adjusted_base_h + (val * scale)
                
                pt_coord[u_ax] = grid_u[iu]
                pt_coord[v_ax] = grid_v[iv]
                pt_coord[axis_idx] = h_val
                
                out_points.InsertNextPoint(pt_coord[0], pt_coord[1], pt_coord[2])
                scalars.SetValue(idx, val)
                idx += 1

        poly.SetPoints(out_points)
        poly.GetPointData().SetScalars(scalars)

        if poly.GetNumberOfPolys() == 0:
            cells = vtk.vtkCellArray()
            for iu in range(self.grid_res_u - 1):
                for iv in range(self.grid_res_v - 1):
                    p0 = iu * self.grid_res_v + iv
                    p1 = (iu + 1) * self.grid_res_v + iv
                    p2 = (iu + 1) * self.grid_res_v + (iv + 1)
                    p3 = iu * self.grid_res_v + (iv + 1)
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, p0)
                    quad.GetPointIds().SetId(1, p1)
                    quad.GetPointIds().SetId(2, p2)
                    quad.GetPointIds().SetId(3, p3)
                    cells.InsertNextCell(quad)
            poly.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(-1.0 if is_diff else 0.0, 1.0)
        mapper.SetLookupTable(lut)

        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetRepresentationToSurface()
        prop.SetInterpolationToPhong()
        prop.LightingOn()

    def reset_camera_to_full_volume(self):
        """Calculates total coordinate extents covering both the particles 
        and the heightmap peak projections, forcing camera focal boundaries."""
        current_scale = self.height_slider.value() / 10.0
        axis_index = self.axis_dropdown.currentIndex()
        
        total_min = list(self.min_coords)
        total_max = list(self.max_coords)
        
        base_h_plane = self.max_coords[axis_index] + 1.0
        
        if self.mode_dropdown.currentIndex() == 0:
            max_height = base_h_plane + (1.0 * current_scale)
        else:
            min_diff = self.grid_data_diff.min()
            z_shift = abs(min_diff) if min_diff < 0 else 0.0
            max_height = base_h_plane + (z_shift * current_scale) + (1.0 * current_scale)
            
        total_max[axis_index] = max_height
        
        self.renderer.ResetCamera(
            total_min[0], total_max[0],
            total_min[1], total_max[1],
            total_min[2], total_max[2]
        )
        self.renderer.ResetCameraClippingRange()
        if hasattr(self, 'vtkWidget') and self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

    def handle_axis_switch(self, axis_index):
        grid_A = self.compute_axis_averages(self.weights_A, axis_index)
        grid_B = self.compute_axis_averages(self.weights_B, axis_index)
        grid_diff = grid_A - grid_B

        max_val_A = grid_A.max() if grid_A.max() != 0 else 1.0
        max_val_B = grid_B.max() if grid_B.max() != 0 else 1.0
        max_val_diff = max(abs(grid_diff.min()), abs(grid_diff.max())) if max(grid_diff.min(), grid_diff.max()) != 0 else 1.0

        self.grid_data_A = grid_A / max_val_A
        self.grid_data_B = grid_B / max_val_B
        self.grid_data_diff = grid_diff / max_val_diff

        self.handle_geometry_update()
        self.reset_camera_to_full_volume()

    def handle_geometry_update(self):
        current_scale = self.height_slider.value() / 10.0
        self.height_label.setText(f"<b>Height Scale: {current_scale:.1f}x</b>")
        axis_index = self.axis_dropdown.currentIndex()

        self.update_heightmap_geometry(self.grid_data_A, self.surface_points_A, self.poly_A, self.actor_A, self.lut_A, current_scale, axis_index, is_diff=False)
        self.update_heightmap_geometry(self.grid_data_B, self.surface_points_B, self.poly_B, self.actor_B, self.lut_B, current_scale, axis_index, is_diff=False)
        self.update_heightmap_geometry(self.grid_data_diff, self.surface_points_diff, self.poly_diff, self.actor_diff, self.lut_diff, current_scale, axis_index, is_diff=True)

        self.handle_mode_switch(self.mode_dropdown.currentIndex())

    def handle_mode_switch(self, index):
        if index == 0:
            self.actor_A.SetVisibility(True)
            self.actor_B.SetVisibility(True)
            self.actor_diff.SetVisibility(False)
            
            self.opacity_slider_B.setVisible(True)
            self.opacity_label_B.setVisible(True)
            self.opacity_label_A.setText(f"<b>Volume A Opacity: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            
            self.actor_A.GetProperty().SetOpacity(self.opacity_slider_A.value() / 100.0)
            self.actor_B.GetProperty().SetOpacity(self.opacity_slider_B.value() / 100.0)
        else:
            self.actor_A.SetVisibility(False)
            self.actor_B.SetVisibility(False)
            self.actor_diff.SetVisibility(True)
            
            self.opacity_slider_B.setVisible(False)
            self.opacity_label_B.setVisible(False)
            self.opacity_label_A.setText(f"<b>Diff Opacity: {self.opacity_slider_A.value()/100.0:.2f}</b>")
            
            self.actor_diff.GetProperty().SetOpacity(self.opacity_slider_A.value() / 100.0)

        self.reset_camera_to_full_volume()

    def handle_opacity_A_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_A.setText(f"<b>Volume A Opacity: {alpha:.2f}</b>")
            if self.actor_A: self.actor_A.GetProperty().SetOpacity(alpha)
        else:
            self.opacity_label_A.setText(f"<b>Diff Opacity: {alpha:.2f}</b>")
            if self.actor_diff: self.actor_diff.GetProperty().SetOpacity(alpha)
        self.vtkWidget.GetRenderWindow().Render()

    def handle_opacity_B_change(self, value):
        alpha = value / 100.0
        if self.mode_dropdown.currentIndex() == 0:
            self.opacity_label_B.setText(f"<b>Volume B Opacity: {alpha:.2f}</b>")
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
    viewer = FullHeightmapViewer(center_coords, w_A, w_B)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()