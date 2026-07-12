import os
import ast
import numpy as np
import pandas as pd
import sys
import xml.etree.ElementTree as ET

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ===================================================================
# 1. CORE DATA COMPILING LAYER (SYNCHRONIZED WITH THE 5% GENERATOR)
# ===================================================================
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
    
    # MATCHES GENERATOR EXPORTER: Force only ~5% of edges to contain distinct delta differences
    np.random.seed(42)
    different_mask = np.random.rand(num_edges) < 0.05 

    weights_A = np.zeros(num_edges)
    weights_B = np.zeros(num_edges)

    for i, edge in enumerate(edge_list):
        p1, p2 = np.array(center_coords[edge[0]]), np.array(center_coords[edge[1]])
        distance = np.linalg.norm(p1 - p2) or 1e-4
        base_weight = 75.0 / distance

        if not different_mask[i]:
            # ~95% of edges get identical noise coefficients
            shared_noise = np.random.uniform(0.90, 1.10)
            weights_A[i] = base_weight * shared_noise
            weights_B[i] = base_weight * shared_noise
        else:
            # ~5% of edges diverge realistically
            weights_A[i] = base_weight * np.random.uniform(0.85, 1.15)
            weights_B[i] = base_weight * np.random.uniform(0.85, 1.15)

    diff_weights = weights_A - weights_B
    return points, edge_list, diff_weights

# ===================================================================
# 2. DYNAMIC LOOKUP PARSING UTILITY
# ===================================================================
def parse_xml_colormap(xml_path):
    if not os.path.exists(xml_path):
        print(f"[Warning] Colormap XML not found at {xml_path}. Falling back to default grading.")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        cmap_node = root.find(".//ColorMap") if root.tag != "ColorMap" else root
        if cmap_node is None:
            print("[Error] No <ColorMap> tag found in XML structure.")
            return None

        control_points = []
        for point in cmap_node.findall("Point"):
            x = float(point.get("x"))
            r = float(point.get("r"))
            g = float(point.get("g"))
            b = float(point.get("b"))
            control_points.append((x, r, g, b))
            
        control_points.sort(key=lambda p: p[0])
        return control_points
    except Exception as e:
        print(f"[Error Parsing XML Colormap File]: {e}")
        return None

# ===================================================================
# 3. INTERFACE LAYER WITH FLIPPED LEGEND SIDEBAR & VALUE ANCHORS
# ===================================================================
class ComparativeViewer(QMainWindow):
    def __init__(self, points, edge_list, diff_weights):
        super().__init__()
        self.setWindowTitle("Steel Beads Contact Networks - 5% Delta View")
        self.setGeometry(100, 100, 1200, 800)

        # ----------------------------------------------------------------
        # SPECIFIED PATHS
        # ----------------------------------------------------------------
        colormap_xml_path = r"colormap\blue-orange-div.xml"  
        colormap_png_path = r"colormap\blue-orange-div.png"
        # ----------------------------------------------------------------

        self.points = points
        self.edge_list = edge_list

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel(f"<b>Active Colormap:</b> {os.path.basename(colormap_xml_path)} | <i>Showing ~5% Network Variations</i>"))
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        # Build clean raw geometric line data structure arrays
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("DifferenceScalars")

        visible_edges_count = 0
        for i, edge in enumerate(self.edge_list):
            val = diff_weights[i]
            
            # STRICT FILTER: Exclude edge entirely if the difference is exactly 0
            if abs(val) < 1e-7: 
                continue
                
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            scalars.InsertNextValue(val)
            visible_edges_count += 1
            
        polydata.SetLines(lines)
        polydata.GetCellData().SetScalars(scalars)
        print(f"[Render Pipeline] Rendering {visible_edges_count} mutated force links.")

        # Build lookup table dynamically from target file path
        lut_xml = vtk.vtkLookupTable()
        lut_xml.SetNumberOfTableValues(256)
        lut_xml.Build()

        parsed_points = parse_xml_colormap(colormap_xml_path)
        
        if parsed_points:
            xp = [pt[0] for pt in parsed_points]
            r_vals = [pt[1] for pt in parsed_points]
            g_vals = [pt[2] for pt in parsed_points]
            b_vals = [pt[3] for pt in parsed_points]

            for i in range(256):
                t = i / 255.0
                r = np.interp(t, xp, r_vals)
                g = np.interp(t, xp, g_vals)
                b = np.interp(t, xp, b_vals)
                lut_xml.SetTableValue(i, r, g, b, 1.0)
        else:
            for i in range(256):
                t = i / 255.0
                if t < 0.5:
                    f = t * 2.0
                    lut_xml.SetTableValue(i, f, f, 1.0, 1.0)
                else:
                    f = 2.0 * (1.0 - t)
                    lut_xml.SetTableValue(i, 1.0, f, f, 1.0)

        min_delta = float(diff_weights.min()) if len(diff_weights) else 0.0
        max_delta = float(diff_weights.max()) if len(diff_weights) else 0.0
        max_bound = max(abs(min_delta), abs(max_delta))

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("DifferenceScalars")
        mapper.SetScalarRange(-max_bound, max_bound)
        mapper.SetLookupTable(lut_xml)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        property_obj = actor.GetProperty()
        property_obj.LightingOff()         
        property_obj.SetLineWidth(1.0)     

        # Core 3D Base Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.09, 0.1)
        self.renderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # ----------------------------------------------------------------
        # FLIPPED VERTICAL LEGEND OVERLAY SIDEBAR
        # ----------------------------------------------------------------
        if os.path.exists(colormap_png_path):
            png_reader = vtk.vtkPNGReader()
            png_reader.SetFileName(colormap_png_path)
            
            flipper = vtk.vtkImageFlip()
            flipper.SetInputConnection(png_reader.GetOutputPort())
            flipper.SetFilteredAxis(1) 
            flipper.Update()

            img_actor = vtk.vtkImageActor()
            img_actor.GetMapper().SetInputConnection(flipper.GetOutputPort())

            self.overlay_renderer = vtk.vtkRenderer()
            self.renderer.SetLayer(0)              
            self.overlay_renderer.SetLayer(1)      
            self.vtkWidget.GetRenderWindow().SetNumberOfLayers(2)

            # API FIX: Proper runtime Boolean flag toggle configuration
            self.overlay_renderer.Transparent()
            
            self.overlay_renderer.SetViewport(0.91, 0.15, 0.94, 0.85)
            self.overlay_renderer.AddActor(img_actor)
            self.overlay_renderer.InteractiveOff()
            self.vtkWidget.GetRenderWindow().AddRenderer(self.overlay_renderer)

            # ------------------------------------------------------------
            # OVERLAY VALUE TEXT GENERATOR
            # ------------------------------------------------------------
            self.text_renderer = vtk.vtkRenderer()
            self.text_renderer.SetLayer(1) 
            self.text_renderer.SetViewport(0.0, 0.0, 1.0, 1.0) 
            self.text_renderer.Transparent()
            self.text_renderer.InteractiveOff()

            def create_text_actor(text_string, x_pos, y_pos):
                t_actor = vtk.vtkTextActor()
                t_actor.SetInput(text_string)
                t_prop = t_actor.GetTextProperty()
                t_prop.SetFontFamilyToArial()
                t_prop.SetFontSize(14)
                t_prop.BoldOn()
                t_prop.SetColor(0.9, 0.9, 0.9)
                t_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                t_actor.GetPositionCoordinate().SetValue(x_pos, y_pos)
                return t_actor

            max_label = create_text_actor(f"Max: {max_bound:+.4f}", 0.83, 0.84)
            min_label = create_text_actor(f"Min: {-max_bound:+.4f}", 0.83, 0.14)

            self.text_renderer.AddActor(max_label)
            self.text_renderer.AddActor(min_label)
            self.vtkWidget.GetRenderWindow().AddRenderer(self.text_renderer)

        # Force camera bounds checks
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

        self.vtkWidget.Initialize()
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtkWidget.GetRenderWindow().Render()

# ===================================================================
# 4. RUNTIME Entry
# ===================================================================
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