import os
import sys
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
)
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


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


class DynamicForceChainVTKViewer(QMainWindow):
    """
    3D Force Chain Viewer with Dynamic Rescaling Edge Color Mapping:
    - Nodes: Solid White spheres
    - Edges: Dynamically colormapped from Deep Blue (Lowest visible edge weight)
             to Bright Red (Highest visible edge weight) among active edges.
    """
    def __init__(self, coords, weights):
        super().__init__()
        self.setWindowTitle("3D Force Chains - Solid White Nodes & Dynamic Edge Force Rescaling")
        self.setGeometry(100, 100, 1100, 850)

        self.coords = coords
        self.raw_weights = weights
        self.num_nodes = len(weights)

        # Build Graph & MST
        self.precompute_mst_graph(k_neighbors=8)

        # UI Layout Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control Panel
        control_layout = QHBoxLayout()
        self.slider_label = QLabel("Weight Threshold Percentile: 65%")
        self.slider_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(65)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.on_slider_changed)

        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.threshold_slider)
        main_layout.addLayout(control_layout)

        # VTK Setup
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.10)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.Initialize()

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(self.style)

        # Setup Persistent Pipeline
        self.setup_vtk_pipeline()

        # Initial Display Update
        self.update_visualization(threshold_percentile=65)
        self.renderer.ResetCamera()

    def precompute_mst_graph(self, k_neighbors=8):
        """Constructs spatial k-NN graph and computes global Maximum Spanning Tree."""
        print(f"[Processing] Precomputing MST backbone for {self.num_nodes} nodes...")
        tree = KDTree(self.coords)

        self.G = nx.Graph()
        for i in range(self.num_nodes):
            self.G.add_node(i, pos=self.coords[i], weight=self.raw_weights[i])

        for i in range(self.num_nodes):
            distances, indices = tree.query(self.coords[i], k=k_neighbors + 1)
            for dist, j in zip(distances[1:], indices[1:]):
                avg_node_w = (self.raw_weights[i] + self.raw_weights[j]) / 2.0
                self.G.add_edge(i, j, weight=avg_node_w)

        self.mst = nx.maximum_spanning_tree(self.G, weight='weight')

    def setup_vtk_pipeline(self):
        """Creates Persistent LUTs, Mappers, and Actors."""
        # --- Node Pipeline (Solid White Spheres) ---
        self.nodes_polydata = vtk.vtkPolyData()

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(0.45)
        sphere_source.SetThetaResolution(12)
        sphere_source.SetPhiResolution(12)

        self.glyph = vtk.vtkGlyph3D()
        self.glyph.SetSourceConnection(sphere_source.GetOutputPort())
        self.glyph.SetInputData(self.nodes_polydata)
        self.glyph.SetScaleModeToDataScalingOff()

        self.nodes_mapper = vtk.vtkPolyDataMapper()
        self.nodes_mapper.SetInputConnection(self.glyph.GetOutputPort())
        self.nodes_mapper.ScalarVisibilityOff()  # Solid color rendering for nodes

        self.nodes_actor = vtk.vtkActor()
        self.nodes_actor.SetMapper(self.nodes_mapper)
        self.nodes_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Pure White
        self.renderer.AddActor(self.nodes_actor)

        # --- Edge Pipeline (Dynamic Blue -> Red Lookup Table) ---
        self.edge_lut = vtk.vtkLookupTable()
        self.edge_lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            # Interpolate from Deep Blue (0, 0.2, 1) to Red (1, 0, 0)
            self.edge_lut.SetTableValue(i, t, 0.2 * (1.0 - t), 1.0 - t, 1.0)
        self.edge_lut.Build()

        self.chains_polydata = vtk.vtkPolyData()

        self.chains_mapper = vtk.vtkPolyDataMapper()
        self.chains_mapper.SetInputData(self.chains_polydata)
        self.chains_mapper.SetLookupTable(self.edge_lut)
        self.chains_mapper.SetScalarModeToUseCellData()

        self.chains_actor = vtk.vtkActor()
        self.chains_actor.SetMapper(self.chains_mapper)
        self.chains_actor.GetProperty().SetLineWidth(3.5)
        self.renderer.AddActor(self.chains_actor)

    def update_visualization(self, threshold_percentile):
        """Re-filters MST graph and dynamically rescales edge colors to visible min/max."""
        min_weight_threshold = np.percentile(self.raw_weights, threshold_percentile)

        # Filter MST graph
        pruned_mst = self.mst.copy()
        nodes_to_remove = [n for n, attr in pruned_mst.nodes(data=True) if attr['weight'] < min_weight_threshold]
        pruned_mst.remove_nodes_from(nodes_to_remove)

        active_nodes = list(pruned_mst.nodes())
        if len(active_nodes) == 0 or pruned_mst.number_of_edges() == 0:
            self.nodes_actor.SetVisibility(False)
            self.chains_actor.SetVisibility(False)
            self.vtkWidget.GetRenderWindow().Render()
            return

        # 1. Update Node Positions
        active_coords = self.coords[active_nodes]
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(active_coords, deep=True))

        self.nodes_polydata.SetPoints(pts)
        self.glyph.Update()
        self.nodes_actor.SetVisibility(True)

        # 2. Update Edges & Dynamically Rescale Color Map
        lines = vtk.vtkCellArray()
        line_pts = vtk.vtkPoints()
        edge_weight_list = []

        point_idx = 0
        for u, v, attr in pruned_mst.edges(data=True):
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

        # Compute dynamic bounds based strictly on visible edges above the threshold
        min_edge_w = float(np.min(edge_weights))
        max_edge_w = float(np.max(edge_weights))
        if min_edge_w == max_edge_w:
            max_edge_w += 1e-5  # Prevent division by zero if all visible edges have equal weight

        # Update Lookup Table & Mapper Scalar Ranges to active subset
        self.edge_lut.SetTableRange(min_edge_w, max_edge_w)
        self.edge_lut.Build()
        
        self.chains_mapper.SetScalarRange(min_edge_w, max_edge_w)

        edge_weight_array = numpy_to_vtk(edge_weights, deep=True)
        edge_weight_array.SetName("EdgeWeights")

        self.chains_polydata.SetPoints(line_pts)
        self.chains_polydata.SetLines(lines)
        self.chains_polydata.GetCellData().SetScalars(edge_weight_array)

        self.chains_actor.SetVisibility(True)

        # Trigger redraw
        self.vtkWidget.GetRenderWindow().Render()

    def on_slider_changed(self, value):
        self.slider_label.setText(f"Weight Threshold Percentile: {value}%")
        self.update_visualization(threshold_percentile=value)


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    base_dir = r"C:\Users\nites\Downloads\morsegramvis-main\morsegramvis-main"
    vtp_path = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")

    coords, weights = load_vtp_nodes_and_weights(vtp_path)

    if coords is None or weights is None:
        return

    app = QApplication(sys.argv)
    viewer = DynamicForceChainVTKViewer(coords, weights)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()