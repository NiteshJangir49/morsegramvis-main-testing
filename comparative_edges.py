import os
import ast
import numpy as np
import pandas as pd
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
from PySide6.QtCore import Qt
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

def compute_network_data(particles_csv, contact_vtp):
    if not os.path.exists(particles_csv) or not os.path.exists(contact_vtp):
        print(f"[Error] Missing paths:\nParticles: {particles_csv}\nContacts: {contact_vtp}")
        return None, [], []

    df_part = pd.read_csv(particles_csv)
    points = vtk.vtkPoints()
    center_coords = []
    id_to_idx = {}
    point_counter = 0
    
    for _, row in df_part.iterrows():
        cp_id = int(row['cp_id'])
        try:
            centroid = row['centroid']
            if isinstance(centroid, str):
                centroid = ast.literal_eval(centroid.strip())
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            if np.isnan(cx) or np.isnan(cy) or np.isnan(cz): continue
                
            center_coords.append((cx, cy, cz))
            points.InsertNextPoint(cx, cy, cz)
            id_to_idx[cp_id] = point_counter
            point_counter += 1
        except Exception:
            continue

    fresh_reader = vtk.vtkXMLPolyDataReader()
    fresh_reader.SetFileName(contact_vtp)
    fresh_reader.Update()
    
    contact_poly = fresh_reader.GetOutput()
    if not contact_poly or not contact_poly.GetPoints():
        return None, [], []

    vtp_pts = contact_poly.GetPoints()
    num_contact_pts = vtp_pts.GetNumberOfPoints()

    seen_edges = set()
    edge_list = []
    
    for i in range(num_contact_pts):
        tx, ty, tz = vtp_pts.GetPoint(i)
        dists = [((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2, idx) for idx, (cx, cy, cz) in enumerate(center_coords)]
        dists.sort(key=lambda x: x[0])
        
        if len(dists) >= 2 and dists[0][1] != dists[1][1]:
            edge_pair = tuple(sorted((dists[0][1], dists[1][1])))
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                edge_list.append(edge_pair)

    num_edges = len(edge_list)
    print(f"[Data Pipeline] Extracted {num_edges} total network edge pairs.")
    
    np.random.seed(42)
    different_mask = np.random.rand(num_edges) < 0.05 

    weights_A = np.zeros(num_edges)
    weights_B = np.zeros(num_edges)

    for i, edge in enumerate(edge_list):
        p1, p2 = np.array(center_coords[edge[0]]), np.array(center_coords[edge[1]])
        distance = np.linalg.norm(p1 - p2) or 1e-4
        base_weight = 75.0 / distance

        if not different_mask[i]:
            shared_noise = np.random.uniform(0.90, 1.10)
            weights_A[i] = base_weight * shared_noise
            weights_B[i] = base_weight * shared_noise
        else:
            weights_A[i] = base_weight * np.random.uniform(0.85, 1.15)
            weights_B[i] = base_weight * np.random.uniform(0.85, 1.15)

    diff_weights = weights_A - weights_B
    return points, edge_list, diff_weights

class ComparativeViewer(QMainWindow):
    def __init__(self, points, edge_list, diff_weights):
        super().__init__()
        self.setWindowTitle("DiffSeer view")
        self.setGeometry(100, 100, 1300, 850)

        self.points = points
        self.edge_list = edge_list

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("<b>Visualization Mode:</b>"))
        
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems([
            "Difference View",
            "Identical Edges",
            "Combined View"
        ])
        self.mode_dropdown.currentIndexChanged.connect(self.handle_mode_switch)
        control_layout.addWidget(self.mode_dropdown)
        
        control_layout.addSpacing(30)
        
        self.slider_label = QLabel("<b>Threshold value: 0.00</b>")
        control_layout.addWidget(self.slider_label)
        
        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setMinimum(0)
        self.filter_slider.setMaximum(100)
        self.filter_slider.setValue(0)
        self.filter_slider.setFixedWidth(200)
        self.filter_slider.valueChanged.connect(self.handle_slider_filter)
        control_layout.addWidget(self.filter_slider)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.09, 0.1)

        max_val = max(abs(diff_weights.min()), abs(diff_weights.max())) if len(diff_weights) else 1.0
        if max_val == 0: max_val = 1e-4
        self.normalized_diffs = diff_weights / max_val 

        min_line_width = 0.1  
        max_line_width = 4.0 

        identical_cells = vtk.vtkCellArray()
        
        self.varied_actors = []
        self.varied_magnitudes = []
        
        varied_count = 0
        identical_count = 0

        for i, edge in enumerate(edge_list):
            norm_val = self.normalized_diffs[i]
            
            if abs(norm_val) < 1e-6:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, edge[0])
                line.GetPointIds().SetId(1, edge[1])
                identical_cells.InsertNextCell(line)
                identical_count += 1
                continue

            magnitude = abs(norm_val)
            fractional_width = min_line_width + (magnitude * (max_line_width - min_line_width))

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            
            single_cell_array = vtk.vtkCellArray()
            single_cell_array.InsertNextCell(line)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(single_cell_array)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            prop = actor.GetProperty()
            prop.LightingOff()
            prop.SetLineWidth(fractional_width)
            
            if norm_val < 0:
                prop.SetColor(0.9, 0.1, 0.1)  
            else:
                prop.SetColor(0.1, 0.5, 0.9)  

            self.renderer.AddActor(actor)
            self.varied_actors.append(actor)
            self.varied_magnitudes.append(magnitude)
            varied_count += 1

        self.identical_actor = vtk.vtkActor()
        if identical_count > 0:
            polydata_id = vtk.vtkPolyData()
            polydata_id.SetPoints(points)
            polydata_id.SetLines(identical_cells)

            mapper_id = vtk.vtkPolyDataMapper()
            mapper_id.SetInputData(polydata_id)

            self.identical_actor.SetMapper(mapper_id)
            prop_id = self.identical_actor.GetProperty()
            prop_id.LightingOff()
            prop_id.SetColor(0.4, 0.45, 0.5) 
            prop_id.SetLineWidth(0.6)        
            self.renderer.AddActor(self.identical_actor)

        print(f"[Render Engine] Parsed {varied_count} mutations and {identical_count} static links.")
        
        self.handle_mode_switch(0)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtkWidget.GetRenderWindow().Render()

    def handle_mode_switch(self, index):
        if index == 1:
            self.filter_slider.setEnabled(False)
            self.slider_label.setText("<b>Threshold value: N/A</b>")
        else:
            self.filter_slider.setEnabled(True)
            current_thresh = self.filter_slider.value() / 100.0
            self.slider_label.setText(f"<b>Threshold value: {current_thresh:.2f}</b>")

        self.apply_visibility_rules()

    def handle_slider_filter(self, value):
        threshold = value / 100.0
        self.slider_label.setText(f"<b>Threshold value: {threshold:.2f}</b>")
        self.apply_visibility_rules()

    def apply_visibility_rules(self):
        mode_idx = self.mode_dropdown.currentIndex()
        threshold = self.filter_slider.value() / 100.0

        show_varied = (mode_idx == 0 or mode_idx == 2)
        show_identical = (mode_idx == 1 or mode_idx == 2)

        for actor, mag in zip(self.varied_actors, self.varied_magnitudes):
            if show_varied and mag >= threshold:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()

        if self.identical_actor:
            if show_identical:
                self.identical_actor.VisibilityOn()
            else:
                self.identical_actor.VisibilityOff()

        if hasattr(self, 'vtkWidget') and self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    particles_file = os.path.join(base_dir, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
    contact_vtp_source = os.path.join(base_dir, "dataset_files", "Steel Beads", "chamf_distance_Steel_Deposition_181_176_251_downsampled_contacts.vtp")

    points, edge_list, diff_weights = compute_network_data(particles_file, contact_vtp_source)
    
    if points is None or len(edge_list) == 0:
        return

    app = QApplication(sys.argv)
    viewer = ComparativeViewer(points, edge_list, diff_weights)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()