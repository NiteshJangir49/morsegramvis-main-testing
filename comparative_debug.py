import os
import ast
import numpy as np
import pandas as pd
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ===================================================================
# DATA ENGINE PIPELINE
# ===================================================================
def load_and_verify_data():
    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    particles_file = os.path.join(base_dir, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
    contact_vtp_source = os.path.join(base_dir, "dataset_files", "Steel Beads", "chamf_distance_Steel_Deposition_181_176_251_downsampled_contacts.vtp")

    if not os.path.exists(particles_file) or not os.path.exists(contact_vtp_source):
        print("[Fatal Setup Error] File paths cannot be resolved.")
        return None, [], []

    df_part = pd.read_csv(particles_file)
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
            center_coords.append((cx, cy, cz))
            points.InsertNextPoint(cx, cy, cz)
            id_to_idx[cp_id] = point_counter
            point_counter += 1
        except Exception:
            continue

    fresh_reader = vtk.vtkXMLPolyDataReader()
    fresh_reader.SetFileName(contact_vtp_source)
    fresh_reader.Update()
    
    contact_poly = fresh_reader.GetOutput()
    vtp_pts = contact_poly.GetPoints()
    edge_list = []
    seen_edges = set()
    
    for i in range(vtp_pts.GetNumberOfPoints()):
        tx, ty, tz = vtp_pts.GetPoint(i)
        dists = [((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2, idx) for idx, (cx, cy, cz) in enumerate(center_coords)]
        dists.sort(key=lambda x: x[0])
        if len(dists) >= 2 and dists[0][1] != dists[1][1]:
            edge_pair = tuple(sorted((dists[0][1], dists[1][1])))
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                edge_list.append(edge_pair)

    # Generate quick default weights array
    np.random.seed(42)
    weights = np.random.uniform(10.0, 50.0, len(edge_list))
    return points, edge_list, weights

# ===================================================================
# LIVE RUNTIME DEBUGGER AND APPLICATION WRAPPER
# ===================================================================
class WindowsVTKDebuggerApp(QMainWindow):
    def __init__(self, points, edge_list, weights):
        super().__init__()
        self.setWindowTitle("Windows Graphics Interop Diagnostics Panel")
        self.setGeometry(100, 100, 1000, 700)

        self.points = points
        self.edge_list = edge_list
        self.weights = weights

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Debug Action Toolbar Layout 
        btn_layout = QHBoxLayout()
        
        self.btn_audit = QPushButton("Print Canvas Memory Audit")
        self.btn_audit.clicked.connect(self.run_canvas_memory_audit)
        btn_layout.addWidget(self.btn_audit)

        self.btn_native = QPushButton("Launch Pure Native Mode (No PySide)")
        self.btn_native.clicked.connect(self.launch_native_window_fallback)
        btn_layout.addWidget(self.btn_native)

        layout.addLayout(btn_layout)

        # Embedded VTK Context Layer
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.05, 0.05, 0.07)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.build_mesh_pipeline()

    def build_mesh_pipeline(self):
        """Assembles the 3D pipeline network explicitly into the renderer."""
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.points)
        
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("Weights")

        for i, edge in enumerate(self.edge_list):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            scalars.InsertNextValue(self.weights[i])

        polydata.SetLines(lines)
        polydata.GetCellData().SetScalars(scalars)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(1.5)  # Intentionally large radius for easy identification
        tube.SetNumberOfSides(6)
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarRange(self.weights.min(), self.weights.max())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetLighting(True)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()

    def run_canvas_memory_audit(self):
        """Dumps internal pipeline properties directly to the console window."""
        print("\n" + "="*50)
        print("          LIVE WINDOWS GRAPHICS CANVAS AUDIT")
        print("="*50)
        
        # 1. Pipeline check
        num_actors = self.ren.GetActors().GetNumberOfItems()
        print(f" -> Active Actors registered in Renderer: {num_actors}")
        
        if num_actors > 0:
            self.actor.GetMapper().Update()
            b = self.actor.GetBounds()
            print(f" -> Actor Physical Spatial Bounds: X:[{b[0]:.1f}:{b[1]:.1f}] | Y:[{b[2]:.1f}:{b[3]:.1f}]")
            print(f" -> Actor Visibility Setting Flag: {self.actor.GetVisibility()}")
            
        # 2. Camera tracking diagnostics
        cam = self.ren.GetActiveCamera()
        print(f" -> Camera Focal Point Position: {[round(x,1) for x in cam.GetFocalPoint()]}")
        print(f" -> Camera Lens Position Vector: {[round(x,1) for x in cam.GetPosition()]}")
        print(f" -> Camera Clipping Slices Range: {[round(x,1) for x in cam.GetClippingRange()]}")
        
        # 3. Hardware handle verification
        win = self.vtkWidget.GetRenderWindow()
        print(f" -> Render Window Mapping Context Valid: {bool(win.GetNeverRendered() == 0)}")
        print(f" -> Graphics Card Driver String: {win.GetReportCapabilitiesString().splitlines()[0] if win.GetReportCapabilitiesString() else 'Unknown OpenGL Subsystem'}")
        print("="*50)

    def launch_native_window_fallback(self):
        """Bypasses PySide components completely to open a raw, native Win32 window canvas."""
        print("\n[Diagnostic Fallback] Launching clean native Win32 render pipe...")
        
        native_win = vtk.vtkRenderWindow()
        native_win.SetWindowName("Direct VTK Native Pipeline Driver Window")
        native_win.SetSize(800, 600)
        
        native_ren = vtk.vtkRenderer()
        native_ren.SetBackground(0.05, 0.08, 0.1)
        native_win.AddRenderer(native_ren)
        
        # Duplicate the network actor handles natively
        native_ren.AddActor(self.actor)
        native_ren.ResetCamera()
        native_ren.ResetCameraClippingRange()
        
        native_iren = vtk.vtkRenderWindowInteractor()
        native_iren.SetRenderWindow(native_win)
        native_iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        native_win.Render()
        print(" -> Direct window rendering engine active. Look for a separate window frame pop-up!")
        native_iren.Initialize()
        # Non-blocking start wrapper check
        native_win.Render()

    def showEvent(self, event):
        super().showEvent(event)
        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtkWidget.Start()
        self.vtkWidget.GetRenderWindow().Render()

def main():
    points, edge_list, weights = load_and_verify_data()
    if points is None or len(edge_list) == 0:
        return

    app = QApplication(sys.argv)
    viewer = WindowsVTKDebuggerApp(points, edge_list, weights)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()