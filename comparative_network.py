import os
import sys
import numpy as np
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def load_vtp_network_scalars(vtp_path_A, vtp_path_B):
    if not os.path.exists(vtp_path_A) or not os.path.exists(vtp_path_B):
        print(f"[Error] VTP path missing:\nA: {vtp_path_A}\nB: {vtp_path_B}")
        return None, None

    reader_A = vtk.vtkXMLPolyDataReader()
    reader_A.SetFileName(vtp_path_A)
    reader_A.Update()
    poly_A = reader_A.GetOutput()

    reader_B = vtk.vtkXMLPolyDataReader()
    reader_B.SetFileName(vtp_path_B)
    reader_B.Update()
    poly_B = reader_B.GetOutput()

    scalars_A = poly_A.GetPointData().GetScalars()
    scalars_B = poly_B.GetPointData().GetScalars()

    if not scalars_A or not scalars_B:
        print("[Error] VTP polydata missing point scalar arrays.")
        return None, None

    weights_A = vtk_to_numpy(scalars_A).astype(np.float32)
    weights_B = vtk_to_numpy(scalars_B).astype(np.float32)

    return weights_A, weights_B


class ZoomableVolumeMatrixComparator(QMainWindow):
    def __init__(self, weights_A, weights_B):
        super().__init__()
        self.setGeometry(100, 100, 1100, 950)

        self.weights_A = weights_A
        self.weights_B = weights_B
        self.num_nodes = len(self.weights_A)

        max_val = max(np.max(self.weights_A), np.max(self.weights_B)) or 1.0
        self.norm_A = np.clip(self.weights_A / max_val, 0.0, 1.0)
        self.norm_B = np.clip(self.weights_B / max_val, 0.0, 1.0)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        main_layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.10)
        
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.Initialize()
        
        self.style = vtk.vtkInteractorStyleImage()
        self.vtkWidget.SetInteractorStyle(self.style)

        # Grayscale lookup table
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.SetTableRange(0.0, 1.0)
        for i in range(256):
            val = i / 255.0
            self.lut.SetTableValue(i, val, val, val, 1.0)
        self.lut.Build()

        self.image_data = vtk.vtkImageData()
        self.image_actor = vtk.vtkImageActor()
        self.renderer.AddActor(self.image_actor)

        self.build_highres_matrix_texture()
        self.reset_camera()

    def build_highres_matrix_texture(self):
        N = self.num_nodes
        sub_res = 16  
        
        # 1. Symmetric outer product matrices
        mat_A = (self.norm_A[:, None] + self.norm_A[None, :]) / 2.0
        np.fill_diagonal(mat_A, self.norm_A)

        mat_B = (self.norm_B[:, None] + self.norm_B[None, :]) / 2.0
        np.fill_diagonal(mat_B, self.norm_B)

        # 2. Tile to high-res sub-pixel grid
        grid_B = np.repeat(np.repeat(mat_B, sub_res, axis=0), sub_res, axis=1)
        grid_A = np.repeat(np.repeat(mat_A, sub_res, axis=0), sub_res, axis=1)
        
        # 3. Inner square mask (Alper et al., 2013 nested visual encoding)
        mask = np.zeros((sub_res, sub_res), dtype=bool)
        m_start, m_end = int(sub_res * 0.25), int(sub_res * 0.75)
        mask[m_start:m_end, m_start:m_end] = True
        
        tile_mask = np.tile(mask, (N, N))
        texture = np.where(tile_mask, grid_A, grid_B)

        # --- FIX: Flip array vertically to match VTK's bottom-left origin ---
        texture = np.flipud(texture)

        h, w = texture.shape
        self.image_data.SetDimensions(w, h, 1)
        self.image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

        vtk_arr = numpy_to_vtk(texture.astype(np.float32).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        self.image_data.GetPointData().SetScalars(vtk_arr)

        img_colors = vtk.vtkImageMapToColors()
        img_colors.SetInputData(self.image_data)
        img_colors.SetLookupTable(self.lut)
        img_colors.Update()

        self.image_actor.GetMapper().SetInputConnection(img_colors.GetOutputPort())
        self.image_actor.GetProperty().SetInterpolationTypeToNearest()

        self.vtkWidget.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    node_vtp_A = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")
    node_vtp_B = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_B.vtp")

    weights_A, weights_B = load_vtp_network_scalars(node_vtp_A, node_vtp_B)

    if weights_A is None or len(weights_A) == 0:
        return

    app = QApplication(sys.argv)
    viewer = ZoomableVolumeMatrixComparator(weights_A, weights_B)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()