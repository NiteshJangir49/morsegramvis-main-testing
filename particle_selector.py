import os
import sys
import numpy as np
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton
)
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def load_vtp_polydata_and_weights(vtp_path):
    """Parses full PolyData mesh, 3D centroid coordinates, and scalar weights from a VTP file."""
    if not os.path.exists(vtp_path):
        print(f"[Error] File not found: {vtp_path}")
        return None, None, None

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    scalars = polydata.GetPointData().GetScalars()

    if points is None or scalars is None:
        print("[Error] Missing points or scalar array in VTP file.")
        return None, None, None

    coords = vtk_to_numpy(points.GetData()).astype(np.float32)
    weights = vtk_to_numpy(scalars).astype(np.float32)

    return polydata, coords, weights


class ParticleSelectionViewer(QMainWindow):
    """
    Dedicated Interactive Particle Selection Interface:
    - Renders original VTP particle boundary mesh with transparency.
    - Displays interactive centroid markers.
    - Mouse Hover -> Highlights particle center in Cyan/Neon Blue.
    - Mouse Click -> Toggles particle selection state in Bright Yellow.
    """
    def __init__(self, polydata, coords, weights):
        super().__init__()
        self.setWindowTitle("Granular Ensemble - Interactive Particle Selection")
        self.setGeometry(100, 100, 1100, 850)

        self.polydata = polydata
        self.coords = coords
        self.raw_weights = weights
        self.num_nodes = len(weights)

        # Selection & Hover State Tracking
        self.selected_node_indices = set()
        self.hovered_node_idx = -1

        # Color Definitions (RGB 0-255)
        self.COLOR_NORMAL = (200, 200, 200)   # Light Gray
        self.COLOR_HOVER = (0, 230, 255)      # Vibrant Cyan / Neon Blue
        self.COLOR_SELECTED = (255, 230, 0)   # Bright Yellow

        # Central Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Bar Controls
        control_layout = QHBoxLayout()
        
        self.selection_status_label = QLabel("Selected Particles: 0")
        self.selection_status_label.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 14px;")

        self.hover_info_label = QLabel("Hovered Particle: None")
        self.hover_info_label.setStyleSheet("font-weight: bold; color: #00e6ff; font-size: 14px;")

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.clear_btn.clicked.connect(self.clear_selection)

        control_layout.addWidget(self.selection_status_label)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.hover_info_label)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.clear_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Main VTK Selection Viewport
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.06, 0.06, 0.08)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.SetInteractorStyle(self.style)

        # Event Observers for Mouse Interactions
        self.vtk_widget.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.vtk_widget.AddObserver("LeftButtonPressEvent", self.on_mouse_click)

        # VTK Point Picker
        self.picker = vtk.vtkPointPicker()
        self.picker.SetTolerance(0.015)

        # Setup 3D Pipelines
        self.setup_scene_pipeline()
        self.update_centroid_colors()

        self.renderer.ResetCamera()

    def setup_scene_pipeline(self):
        """Builds original particle mesh and pickable centroid markers."""
        # 1. Original VTP Mesh Boundaries with Opacity
        mapper_mesh = vtk.vtkPolyDataMapper()
        mapper_mesh.SetInputData(self.polydata)
        mapper_mesh.ScalarVisibilityOff()

        self.actor_mesh = vtk.vtkActor()
        self.actor_mesh.SetMapper(mapper_mesh)
        self.actor_mesh.GetProperty().SetColor(0.75, 0.75, 0.85)
        self.actor_mesh.GetProperty().SetOpacity(0.30)
        self.renderer.AddActor(self.actor_mesh)

        # 2. Interactive Centroid Markers
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(self.coords, deep=True))

        self.center_colors = vtk.vtkUnsignedCharArray()
        self.center_colors.SetNumberOfComponents(3)
        self.center_colors.SetName("CenterColors")

        self.centers_poly = vtk.vtkPolyData()
        self.centers_poly.SetPoints(pts)

        center_src = vtk.vtkSphereSource()
        center_src.SetRadius(0.28)
        center_src.SetThetaResolution(12)
        center_src.SetPhiResolution(12)

        self.glyph_centers = vtk.vtkGlyph3D()
        self.glyph_centers.SetSourceConnection(center_src.GetOutputPort())
        self.glyph_centers.SetInputData(self.centers_poly)
        self.glyph_centers.SetColorModeToColorByScalar()
        self.glyph_centers.SetScaleModeToDataScalingOff()

        mapper_centers = vtk.vtkPolyDataMapper()
        mapper_centers.SetInputConnection(self.glyph_centers.GetOutputPort())
        mapper_centers.SetScalarModeToUsePointData()

        self.actor_centers = vtk.vtkActor()
        self.actor_centers.SetMapper(mapper_centers)
        self.renderer.AddActor(self.actor_centers)

    def highlight_particle_center(self, point_idx):
        """
        Dedicated function to update hover state and highlight the target 
        particle center with a vibrant cyan color.
        """
        if self.hovered_node_idx == point_idx:
            return  # No state change needed

        self.hovered_node_idx = point_idx

        if point_idx != -1:
            self.hover_info_label.setText(
                f"Hovered Particle: ID {point_idx} (Weight: {self.raw_weights[point_idx]:.3f})"
            )
        else:
            self.hover_info_label.setText("Hovered Particle: None")

        # Refresh centroid color array
        self.update_centroid_colors()

    def update_centroid_colors(self):
        """Refreshes color buffer for normal, hovered, and selected centroids."""
        self.center_colors.Reset()

        for i in range(self.num_nodes):
            if i in self.selected_node_indices:
                # Selected Particle -> Bright Yellow
                self.center_colors.InsertNextTypedTuple(self.COLOR_SELECTED)
            elif i == self.hovered_node_idx:
                # Hovered Particle -> Vibrant Cyan Highlight
                self.center_colors.InsertNextTypedTuple(self.COLOR_HOVER)
            else:
                # Normal Particle -> Light Gray
                self.center_colors.InsertNextTypedTuple(self.COLOR_NORMAL)

        self.centers_poly.GetPointData().SetScalars(self.center_colors)
        self.glyph_centers.Update()
        self.vtk_widget.GetRenderWindow().Render()

    def clear_selection(self):
        """Clears all currently selected particles."""
        self.selected_node_indices.clear()
        self.selection_status_label.setText("Selected Particles: 0")
        self.update_centroid_colors()

    # --- Mouse Interaction Events ---
    def on_mouse_move(self, obj, event):
        """Detects mouse hover over particle centroids and triggers highlight function."""
        click_pos = self.vtk_widget.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        point_idx = self.picker.GetPointId()

        if 0 <= point_idx < self.num_nodes:
            self.highlight_particle_center(point_idx)
        else:
            self.highlight_particle_center(-1)

        self.style.OnMouseMove()

    def on_mouse_click(self, obj, event):
        """Toggles selection state of clicked particle centroid."""
        click_pos = self.vtk_widget.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        point_idx = self.picker.GetPointId()

        if 0 <= point_idx < self.num_nodes:
            if point_idx in self.selected_node_indices:
                self.selected_node_indices.remove(point_idx)
            else:
                self.selected_node_indices.add(point_idx)

            self.selection_status_label.setText(f"Selected Particles: {len(self.selected_node_indices)}")
            self.update_centroid_colors()

        self.style.OnLeftButtonDown()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    base_dir = r"C:\Users\nites\Downloads\morsegramvis-main\morsegramvis-main"
    vtp_path = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")

    polydata, coords, weights = load_vtp_polydata_and_weights(vtp_path)

    if polydata is None or coords is None or weights is None:
        return

    app = QApplication(sys.argv)
    viewer = ParticleSelectionViewer(polydata, coords, weights)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()