import os
import sys
import numpy as np
import pandas as pd

import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QCheckBox, QComboBox, QSplitter, QFrame, QDialog, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QDoubleValidator, QFont
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Matplotlib integration for Matrix Rendering
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.colors as mcolors


# ==============================================================================
# CUSTOM DUAL-HANDLE RANGE SLIDER WIDGET WITH RULER TICKS & VALUE LABELS
# ==============================================================================
class RangeSlider(QWidget):
    valuesChanged = Signal(int, int)

    def __init__(self, minimum=0, maximum=100, low=0, high=100, parent=None):
        super().__init__(parent)
        self.min_val = minimum
        self.max_val = maximum
        self.low_val = low
        self.high_val = high

        self.max_force_val = 1.0

        self.handle_radius = 6.5
        self.active_handle = None

        self.setMinimumHeight(44)
        self.setFixedWidth(145)

    def setMaxForceVal(self, max_val):
        self.max_force_val = max(1e-5, float(max_val))
        self.update()

    def setValues(self, low, high):
        self.low_val = max(self.min_val, min(low, self.max_val))
        self.high_val = max(self.low_val, min(high, self.max_val))
        self.update()
        self.valuesChanged.emit(self.low_val, self.high_val)

    def getValues(self):
        return self.low_val, self.high_val

    def _val_to_pos(self, val):
        width = self.width() - 2 * self.handle_radius
        if self.max_val == self.min_val:
            return self.handle_radius
        return self.handle_radius + (val - self.min_val) / (self.max_val - self.min_val) * width

    def _pos_to_val(self, pos):
        width = self.width() - 2 * self.handle_radius
        if width <= 0:
            return self.min_val
        t = (pos - self.handle_radius) / width
        val = self.min_val + t * (self.max_val - self.min_val)
        if pos <= self.handle_radius:
            return self.min_val
        if pos >= self.width() - self.handle_radius:
            return self.max_val
        return int(round(max(self.min_val, min(val, self.max_val))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        cy = 13.0

        track_rect = QRectF(self.handle_radius, cy - 2.5, w - 2 * self.handle_radius, 5)

        # 1. Background Track
        painter.setBrush(QBrush(QColor("#333333")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 2.5, 2.5)

        # 2. Active Span Between Low and High
        low_x = self._val_to_pos(self.low_val)
        high_x = self._val_to_pos(self.high_val)

        span_rect = QRectF(low_x, cy - 2.5, max(0.0, high_x - low_x), 5)
        painter.setBrush(QBrush(QColor("#00bcd4")))
        painter.drawRoundedRect(span_rect, 2.5, 2.5)

        # 3. Ruler Scale Ticks Below Track
        major_pen = QPen(QColor("#999999"), 1.0)
        minor_pen = QPen(QColor("#555555"), 0.8)

        tick_top = cy + 6.0
        for pct in range(0, 101, 10):
            tx = self._val_to_pos(pct)
            if pct % 50 == 0:
                painter.setPen(major_pen)
                painter.drawLine(int(tx), int(tick_top), int(tx), int(tick_top + 5.5))
            elif pct % 20 == 0:
                painter.setPen(minor_pen)
                painter.drawLine(int(tx), int(tick_top), int(tx), int(tick_top + 3.5))
            else:
                painter.setPen(minor_pen)
                painter.drawLine(int(tx), int(tick_top), int(tx), int(tick_top + 2.0))

        # 4. Numeric Value Labels
        font = QFont()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#aaaaaa"))

        label_y = tick_top + 7.0
        label_h = 13.0

        pos_0 = self._val_to_pos(0)
        pos_50 = self._val_to_pos(50)
        pos_100 = self._val_to_pos(100)

        mid_val = self.max_force_val / 2.0

        rect_0 = QRectF(pos_0 - 15, label_y, 30, label_h)
        painter.drawText(rect_0, Qt.AlignHCenter | Qt.AlignTop, "0")

        rect_50 = QRectF(pos_50 - 20, label_y, 40, label_h)
        painter.drawText(rect_50, Qt.AlignHCenter | Qt.AlignTop, f"{mid_val:.2f}")

        rect_100 = QRectF(pos_100 - 25, label_y, 40, label_h)
        painter.drawText(rect_100, Qt.AlignHCenter | Qt.AlignTop, f"{self.max_force_val:.2f}")

        # 5. Interactive Handles
        for x, color, border in [(low_x, QColor("#00e6ff"), QColor("#ffffff")), 
                                 (high_x, QColor("#ff9800"), QColor("#ffffff"))]:
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(border, 1.2))
            painter.drawEllipse(QRectF(x - self.handle_radius, cy - self.handle_radius, 
                                       2 * self.handle_radius, 2 * self.handle_radius))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.position().x()
            low_x = self._val_to_pos(self.low_val)
            high_x = self._val_to_pos(self.high_val)

            d_low = abs(x - low_x)
            d_high = abs(x - high_x)

            if d_low <= d_high:
                self.active_handle = 'low'
                self.low_val = min(self._pos_to_val(x), self.high_val)
            else:
                self.active_handle = 'high'
                self.high_val = max(self._pos_to_val(x), self.low_val)

            self.update()
            self.valuesChanged.emit(self.low_val, self.high_val)

    def mouseMoveEvent(self, event):
        if self.active_handle is not None:
            x = event.position().x()
            val = self._pos_to_val(x)

            if self.active_handle == 'low':
                self.low_val = min(val, self.high_val)
            elif self.active_handle == 'high':
                self.high_val = max(val, self.low_val)

            self.update()
            self.valuesChanged.emit(self.low_val, self.high_val)

    def mouseReleaseEvent(self, event):
        self.active_handle = None


# ==============================================================================
# DATA LOADING UTILITY
# ==============================================================================
def load_real_and_predicted_data(assembly_id=58, timestep=80):
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
            return None, None, None, None, None, None, None

    df_x = pd.read_csv(pos_x_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'x'])
    df_y = pd.read_csv(pos_y_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'y'])
    df_z = pd.read_csv(pos_z_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'z'])

    df_x['ball_id'] = df_x['ball_id'].astype(int)
    df_y['ball_id'] = df_y['ball_id'].astype(int)
    df_z['ball_id'] = df_z['ball_id'].astype(int)

    df_coords = df_x.merge(df_y, on='ball_id').merge(df_z, on='ball_id')

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

    df_pred = pd.read_csv(pred_npmncf_path)

    df_nodes = df_pred.merge(df_actual[['ball_id', 'node_weight_actual']], on='ball_id', how='inner')
    df_nodes = df_nodes.merge(df_coords, on='ball_id', how='inner')

    ball_ids = df_nodes['ball_id'].values
    num_nodes = len(ball_ids)
    id_to_idx = {bid: idx for idx, bid in enumerate(ball_ids)}

    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float32) * 100.0

    node_weights_a = df_nodes['node_weight_actual'].values.astype(np.float32)
    node_weights_b = df_nodes['predicted_npmncf'].values.astype(np.float32)

    end1_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end1.tab")
    end2_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end2.tab")
    pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_contact_force.csv")
    if not os.path.exists(pred_contact_path):
        pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_predicted_contact_force.csv")
    contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_contact_pairs.csv")
    if not os.path.exists(contact_pairs_path):
        contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_contact_pairs.csv")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    weights_a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    weights_b = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if os.path.exists(end1_path) and os.path.exists(end2_path):
        e1 = pd.read_csv(end1_path, sep=r'\s+', header=None, skiprows=2)
        e2 = pd.read_csv(end2_path, sep=r'\s+', header=None, skiprows=2)
        e1_ids = pd.to_numeric(e1.iloc[:, 1], errors='coerce').dropna().astype(int)
        e2_ids = pd.to_numeric(e2.iloc[:, 1], errors='coerce').dropna().astype(int)
        for u_id, v_id in zip(e1_ids, e2_ids):
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])
    elif os.path.exists(pred_contact_path):
        df_contacts = pd.read_csv(pred_contact_path)
        for _, row in df_contacts.iterrows():
            u_id, v_id = int(row['sender_id']), int(row['receiver_id'])
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])
    elif os.path.exists(contact_pairs_path):
        df_pairs = pd.read_csv(contact_pairs_path)
        for _, row in df_pairs.iterrows():
            u_id, v_id = int(row['sender_id']), int(row['receiver_id'])
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1
                weights_a[u, v] = weights_a[v, u] = min(node_weights_a[u], node_weights_a[v])
                weights_b[u, v] = weights_b[v, u] = min(node_weights_b[u], node_weights_b[v])

    return adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids


# ==============================================================================
# MATRIX VIEW DIALOG
# ==============================================================================
class Fig3dMatrixDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matrix View - Actual vs GNN Predicted")
        self.setGeometry(100, 100, 750, 750)

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.fig.patch.set_facecolor('#111111')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        self.cmap_a = mcolors.LinearSegmentedColormap.from_list(
            'vol_a_cyan_blue',
            [(0.0, 0.85, 0.95), (0.1, 0.45, 0.95), (0.05, 0.15, 0.65), (0.02, 0.05, 0.35)]
        )
        self.cmap_b = mcolors.LinearSegmentedColormap.from_list(
            'vol_b_amber_crimson',
            [(1.0, 0.68, 0.0), (0.95, 0.30, 0.05), (0.75, 0.05, 0.10), (0.40, 0.0, 0.05)]
        )

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

        pad_x = 0.08
        pad_bottom = 0.06
        usable_w = (1.0 - 2 * pad_x) / 2.0
        usable_h = 1.0 - 2 * pad_bottom

        for i in range(N):
            for j in range(N):
                u = selected_nodes[i]
                v = selected_nodes[j]

                w_a = weights_matrix_a.get((u, v), 0.0) if u != v else weights_matrix_a.get((u, u), 0.0)
                w_b = weights_matrix_b.get((u, v), 0.0) if u != v else weights_matrix_b.get((u, u), 0.0)

                x = j
                y = N - 1 - i

                is_diagonal = (i == j)
                bg_color = '#C8C8C8' if is_diagonal else '#EBEBEB'

                rect_bg = patches.Rectangle((x, y), 1.0, 1.0, 
                                            facecolor=bg_color, edgecolor='#A0A0A0', linewidth=0.6)
                self.ax.add_patch(rect_bg)

                if w_a > 0 or w_b > 0:
                    norm_a = np.clip((w_a - active_min) / denom, 0.0, 1.0) if w_a > 0 else 0.0
                    norm_b = np.clip((w_b - active_min) / denom, 0.0, 1.0) if w_b > 0 else 0.0

                    height_a = norm_a * usable_h
                    height_b = norm_b * usable_h

                    color_a = self.cmap_a(norm_a)
                    color_b = self.cmap_b(norm_b)

                    if w_a > 0:
                        bar_a_x = x + pad_x
                        bar_a_y = y + pad_bottom
                        bar_a = patches.Rectangle((bar_a_x, bar_a_y), usable_w, height_a, 
                                                  facecolor=color_a, edgecolor='#111111', linewidth=0.3)
                        self.ax.add_patch(bar_a)

                        lum_a = 0.299 * color_a[0] + 0.587 * color_a[1] + 0.114 * color_a[2]
                        txt_color_a = 'black' if lum_a > 0.50 else 'white'
                        self.ax.text(bar_a_x + usable_w / 2.0, bar_a_y + max(height_a / 2.0, 0.12), f"{w_a:.2f}", 
                                     color=txt_color_a, fontsize=font_sz, fontweight='bold',
                                     ha='center', va='center', rotation=90)

                    if w_b > 0:
                        bar_b_x = x + pad_x + usable_w
                        bar_b_y = y + pad_bottom
                        bar_b = patches.Rectangle((bar_b_x, bar_b_y), usable_w, height_b, 
                                                  facecolor=color_b, edgecolor='#111111', linewidth=0.3)
                        self.ax.add_patch(bar_b)

                        lum_b = 0.299 * color_b[0] + 0.587 * color_b[1] + 0.114 * color_b[2]
                        txt_color_b = 'black' if lum_b > 0.50 else 'white'
                        self.ax.text(bar_b_x + usable_w / 2.0, bar_b_y + max(height_b / 2.0, 0.12), f"{w_b:.2f}", 
                                     color=txt_color_b, fontsize=font_sz, fontweight='bold',
                                     ha='center', va='center', rotation=90)

        self.ax.set_xlim(0, N)
        self.ax.set_ylim(0, N)

        self.ax.set_xticks([i + 0.5 for i in range(N)])
        self.ax.set_yticks([N - 1 - i + 0.5 for i in range(N)])
        self.ax.set_xticklabels([str(n) for n in selected_nodes], color='cyan', fontsize=9, rotation=90)
        self.ax.set_yticklabels([str(n) for n in selected_nodes], color='cyan', fontsize=9)

        self.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        self.ax.set_title("Matrix View (Left: Vol A [Cyan-Navy] | Right: Vol B [Amber-Crimson])", color='white', pad=20, fontsize=11)

        self.fig.tight_layout()
        self.canvas.draw()


# ==============================================================================
# MAIN DUAL-PANEL VIEWER
# ==============================================================================
class DualPanelForceChainBuilder(QMainWindow):
    def __init__(self, initial_assembly=58, initial_timestep=80):
        super().__init__()
        self.current_assembly = initial_assembly
        self.current_timestep = initial_timestep

        self.setWindowTitle(f"Parallel Force Chains - (Assembly {self.current_assembly}, Timestep {self.current_timestep})")
        self.setGeometry(50, 50, 1680, 850)

        adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids = load_real_and_predicted_data(
            self.current_assembly, self.current_timestep
        )

        self.adj_matrix = adj_matrix
        self.weights_a = weights_a
        self.weights_b = weights_b
        self.node_weights_a = node_weights_a
        self.node_weights_b = node_weights_b
        self.coords = coords
        self.ball_ids = ball_ids
        self.num_nodes = len(coords)

        self.compute_spatial_bounds()

        all_w = np.concatenate([weights_a[weights_a > 0], weights_b[weights_b > 0]])
        self.global_min_w = 0.0
        self.global_max_w = float(np.max(all_w)) if len(all_w) > 0 else 1.0
        self.active_min_w = 0.0
        self.active_max_w = self.global_max_w

        self.selected_node_indices = set()
        self.hovered_node_idx = -1
        self.active_visible_nodes_left = []
        self.current_selected_nodes_right = []
        self.text_label_actors_right = []

        self.COLOR_NORMAL = (180, 180, 180)
        self.COLOR_HOVER = (0, 255, 255)
        self.COLOR_SELECTED = (255, 230, 0)

        self.build_graphs_from_arrays()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(8)

        # 1. Timestep Text Entry
        control_layout.addWidget(QLabel("<b>Timestep:</b>"))
        self.timestep_input = QLineEdit(str(self.current_timestep))
        self.timestep_input.setFixedWidth(42)
        self.timestep_input.setAlignment(Qt.AlignCenter)
        self.timestep_input.returnPressed.connect(self.handle_timestep_change)
        control_layout.addWidget(self.timestep_input)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(48)
        self.load_btn.setStyleSheet("font-weight: bold; background-color: #005588; color: white;")
        self.load_btn.clicked.connect(self.handle_timestep_change)
        control_layout.addWidget(self.load_btn)

        control_layout.addSpacing(6)

        # 2. Dual Range Slider & Manual Text Input Bounds [0.00, max_force]
        control_layout.addWidget(QLabel("<b>Range:</b>"))
        self.min_weight_input = QLineEdit("0.00")
        self.min_weight_input.setFixedWidth(45)
        self.min_weight_input.setAlignment(Qt.AlignCenter)
        self.min_weight_input.setStyleSheet("""
            QLineEdit {
                font-weight: bold;
                font-size: 9pt;
                color: #00e6ff;
                background-color: #222222;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 1px;
            }
        """)
        self.min_weight_input.returnPressed.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.min_weight_input)

        self.range_slider = RangeSlider(minimum=0, maximum=100, low=0, high=100)
        self.range_slider.setMaxForceVal(self.global_max_w)
        self.range_slider.valuesChanged.connect(self.on_range_changed)
        control_layout.addWidget(self.range_slider)

        self.max_weight_input = QLineEdit(f"{self.global_max_w:.2f}")
        self.max_weight_input.setFixedWidth(45)
        self.max_weight_input.setAlignment(Qt.AlignCenter)
        self.max_weight_input.setStyleSheet("""
            QLineEdit {
                font-weight: bold;
                font-size: 9pt;
                color: #ff9800;
                background-color: #222222;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 1px;
            }
        """)
        self.max_weight_input.returnPressed.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.max_weight_input)

        self.apply_range_btn = QPushButton("Set")
        self.apply_range_btn.setFixedWidth(36)
        self.apply_range_btn.setStyleSheet("font-weight: bold; background-color: #444444; color: white; font-size: 8pt;")
        self.apply_range_btn.clicked.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.apply_range_btn)

        self.update_range_validators()

        control_layout.addSpacing(6)

        # 3. Peak-Neighbor Chains Control
        self.max_neighbor_toggle = QCheckBox("Max-Force Neighbor")
        self.max_neighbor_toggle.setChecked(False)
        self.max_neighbor_toggle.setStyleSheet("font-weight: bold; color: #ff55ff; font-size: 9pt;")
        self.max_neighbor_toggle.toggled.connect(self.on_max_neighbor_toggled)
        control_layout.addWidget(self.max_neighbor_toggle)

        control_layout.addSpacing(6)

        # 4. Camera Sync & ID Toggles
        self.lock_camera_checkbox = QCheckBox("Lock Cam")
        self.lock_camera_checkbox.setChecked(True)
        self.lock_camera_checkbox.setStyleSheet("font-weight: bold; color: #00e6ff; font-size: 9pt;")
        self.lock_camera_checkbox.toggled.connect(self.on_camera_lock_toggled)
        control_layout.addWidget(self.lock_camera_checkbox)

        self.show_ids_checkbox = QCheckBox("IDs (Right)")
        self.show_ids_checkbox.setChecked(True)
        self.show_ids_checkbox.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 9pt;")
        self.show_ids_checkbox.toggled.connect(self.on_show_ids_toggled)
        control_layout.addWidget(self.show_ids_checkbox)

        control_layout.addSpacing(6)

        # 5. Edge Filter Dropdown
        edge_filter_label = QLabel("Edge Filter:")
        edge_filter_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        control_layout.addWidget(edge_filter_label)
        
        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItems(["Default (Union)", "Only from A", "Only from B", "Both (Intersection)"])
        self.edge_mode_combo.setStyleSheet("font-weight: bold; padding: 2px 4px; font-size: 9pt;")
        self.edge_mode_combo.currentIndexChanged.connect(self.on_edge_mode_changed)
        control_layout.addWidget(self.edge_mode_combo)

        control_layout.addSpacing(6)

        # 6. Matrix & Clear Action Buttons
        self.matrix_btn = QPushButton("Matrix View")
        self.matrix_btn.setStyleSheet("font-weight: bold; padding: 3px 8px; background-color: #005588; color: white; font-size: 9pt;")
        self.matrix_btn.clicked.connect(self.open_matrix_window)
        control_layout.addWidget(self.matrix_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("font-weight: bold; padding: 3px 8px; font-size: 9pt;")
        self.clear_btn.clicked.connect(self.clear_selection)
        control_layout.addWidget(self.clear_btn)

        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        self.setup_left_panel()
        self.setup_right_panel()

        self.enable_camera_sync(True)
        self.splitter.setSizes([750, 750])

        self.matrix_dialog = Fig3dMatrixDialog(self)

        self.range_slider.setValues(0, 100)
        self.ren_left.ResetCamera()
        self.sync_clipping_ranges()

    def _setup_orientation_axes_widget(self, interactor):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.04)

        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1.0, 0.25, 0.25)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.25, 1.0, 0.25)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.25, 0.55, 1.0)

        for cap in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            cap.GetCaptionTextProperty().SetFontSize(10)
            cap.GetCaptionTextProperty().BoldOn()
            cap.GetCaptionTextProperty().ShadowOff()

        orient_widget = vtk.vtkOrientationMarkerWidget()
        orient_widget.SetOrientationMarker(axes)
        orient_widget.SetInteractor(interactor)
        orient_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        orient_widget.SetEnabled(1)
        orient_widget.InteractiveOff()

        return orient_widget, axes

    def update_range_validators(self):
        val = QDoubleValidator(0.0, float(self.global_max_w), 2, self)
        val.setNotation(QDoubleValidator.StandardNotation)
        self.min_weight_input.setValidator(val)
        self.max_weight_input.setValidator(val)

    def compute_spatial_bounds(self):
        min_c = np.min(self.coords, axis=0)
        max_c = np.max(self.coords, axis=0)
        self.global_bounds = [min_c[0], max_c[0], min_c[1], max_c[1], min_c[2], max_c[2]]

    def handle_timestep_change(self):
        text = self.timestep_input.text().strip()
        try:
            new_timestep = int(text)
        except ValueError:
            print(f"[Warning] Invalid timestep integer: '{text}'")
            return

        adj_matrix, weights_a, weights_b, node_weights_a, node_weights_b, coords, ball_ids = load_real_and_predicted_data(
            self.current_assembly, new_timestep
        )

        if coords is None or len(weights_a) == 0:
            print(f"[Warning] Failed to load data for Timestep {new_timestep}. Keeping previous state.")
            return

        self.current_timestep = new_timestep
        self.adj_matrix = adj_matrix
        self.weights_a = weights_a
        self.weights_b = weights_b
        self.node_weights_a = node_weights_a
        self.node_weights_b = node_weights_b
        self.coords = coords
        self.ball_ids = ball_ids
        self.num_nodes = len(coords)

        all_w = np.concatenate([weights_a[weights_a > 0], weights_b[weights_b > 0]])
        self.global_max_w = float(np.max(all_w)) if len(all_w) > 0 else 1.0
        self.range_slider.setMaxForceVal(self.global_max_w)
        self.update_range_validators()

        self.compute_spatial_bounds()
        self.build_graphs_from_arrays()
        self.setWindowTitle(f"Parallel Force Chains - (Assembly {self.current_assembly}, Timestep {self.current_timestep})")

        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def build_graphs_from_arrays(self):
        self.all_edges = []
        all_edge_weights = []

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i, j] == 1:
                    wa = self.weights_a[i, j]
                    wb = self.weights_b[i, j]
                    self.all_edges.append((i, j, wa, wb))
                    if wa > 0: all_edge_weights.append(wa)
                    if wb > 0: all_edge_weights.append(wb)

        self.raw_weights = np.array(all_edge_weights, dtype=np.float32)

    def create_lut_a(self):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            r = (0.0 * (1.0 - t) + 0.02 * t)
            g = (0.85 * (1.0 - t)**2 + 0.45 * 2.0 * t * (1.0 - t) + 0.05 * t**2)
            b = (0.95 * (1.0 - t) + 0.35 * t)
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.Build()
        return lut

    def create_lut_b(self):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            r = (1.0 * (1.0 - t) + 0.40 * t)
            g = (0.68 * (1.0 - t)**1.5 + 0.0 * t)
            b = (0.0 * (1.0 - t) + 0.05 * t)
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.Build()
        return lut

    def setup_left_panel(self):
        left_container = QFrame()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)

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
        sphere_src.SetRadius(0.02)
        sphere_src.SetThetaResolution(10)
        sphere_src.SetPhiResolution(10)

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

        self.lut_a = self.create_lut_a()
        self.lut_b = self.create_lut_b()

        self.left_chains_a_polydata = vtk.vtkPolyData()
        self.left_chains_a_mapper = vtk.vtkPolyDataMapper()
        self.left_chains_a_mapper.SetInputData(self.left_chains_a_polydata)
        self.left_chains_a_mapper.SetLookupTable(self.lut_a)
        self.left_chains_a_mapper.SetScalarModeToUseCellData()

        self.left_actor_chains_a = vtk.vtkActor()
        self.left_actor_chains_a.SetMapper(self.left_chains_a_mapper)
        self.left_actor_chains_a.GetProperty().SetLineWidth(1.0)
        self.ren_left.AddActor(self.left_actor_chains_a)

        self.left_chains_b_polydata = vtk.vtkPolyData()
        self.left_chains_b_mapper = vtk.vtkPolyDataMapper()
        self.left_chains_b_mapper.SetInputData(self.left_chains_b_polydata)
        self.left_chains_b_mapper.SetLookupTable(self.lut_b)
        self.left_chains_b_mapper.SetScalarModeToUseCellData()

        self.left_actor_chains_b = vtk.vtkActor()
        self.left_actor_chains_b.SetMapper(self.left_chains_b_mapper)
        self.left_actor_chains_b.GetProperty().SetLineWidth(1.0)
        self.ren_left.AddActor(self.left_actor_chains_b)

        # Floating 3D XYZ Orientation Axes Glyph (Left Viewport)
        self.left_axes_widget, self.left_axes_actor = self._setup_orientation_axes_widget(self.vtk_left)

        self.splitter.addWidget(left_container)

    def setup_right_panel(self):
        right_container = QFrame()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)

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
        sphere_src.SetRadius(0.02)
        sphere_src.SetThetaResolution(10)
        sphere_src.SetPhiResolution(10)

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

        self.right_chains_a_polydata = vtk.vtkPolyData()
        self.right_chains_a_mapper = vtk.vtkPolyDataMapper()
        self.right_chains_a_mapper.SetInputData(self.right_chains_a_polydata)
        self.right_chains_a_mapper.SetLookupTable(self.lut_a)
        self.right_chains_a_mapper.SetScalarModeToUseCellData()

        self.right_actor_chains_a = vtk.vtkActor()
        self.right_actor_chains_a.SetMapper(self.right_chains_a_mapper)
        self.right_actor_chains_a.GetProperty().SetLineWidth(1.0)
        self.ren_right.AddActor(self.right_actor_chains_a)

        self.right_chains_b_polydata = vtk.vtkPolyData()
        self.right_chains_b_mapper = vtk.vtkPolyDataMapper()
        self.right_chains_b_mapper.SetInputData(self.right_chains_b_polydata)
        self.right_chains_b_mapper.SetLookupTable(self.lut_b)
        self.right_chains_b_mapper.SetScalarModeToUseCellData()

        self.right_actor_chains_b = vtk.vtkActor()
        self.right_actor_chains_b.SetMapper(self.right_chains_b_mapper)
        self.right_actor_chains_b.GetProperty().SetLineWidth(1.0)
        self.ren_right.AddActor(self.right_actor_chains_b)

        # Floating 3D XYZ Orientation Axes Glyph (Right Viewport)
        self.right_axes_widget, self.right_axes_actor = self._setup_orientation_axes_widget(self.vtk_right)

        self.splitter.addWidget(right_container)

    def enable_camera_sync(self, enable=True):
        if enable:
            self.ren_right.SetActiveCamera(self.ren_left.GetActiveCamera())
        else:
            independent_cam = vtk.vtkCamera()
            independent_cam.DeepCopy(self.ren_left.GetActiveCamera())
            self.ren_right.SetActiveCamera(independent_cam)

    def sync_clipping_ranges(self):
        self.ren_left.ResetCameraClippingRange(self.global_bounds)
        if self.lock_camera_checkbox.isChecked():
            self.ren_right.ResetCameraClippingRange(self.global_bounds)

    def on_camera_lock_toggled(self, checked):
        self.enable_camera_sync(checked)
        self.sync_clipping_ranges()
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def on_show_ids_toggled(self, checked):
        self.update_right_particle_id_labels()
        self.sync_clipping_ranges()
        self.vtk_right.GetRenderWindow().Render()

    def on_max_neighbor_toggled(self, checked):
        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def on_edge_mode_changed(self, index):
        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def on_range_changed(self, low_percentile, high_percentile):
        min_w = (low_percentile / 100.0) * self.global_max_w
        max_w = (high_percentile / 100.0) * self.global_max_w

        self.min_weight_input.blockSignals(True)
        self.max_weight_input.blockSignals(True)
        self.min_weight_input.setText(f"{min_w:.2f}")
        self.max_weight_input.setText(f"{max_w:.2f}")
        self.min_weight_input.blockSignals(False)
        self.max_weight_input.blockSignals(False)

        self.apply_thresholds(min_w, max_w)

    def on_manual_range_applied(self):
        try:
            min_w = float(self.min_weight_input.text().strip())
        except ValueError:
            min_w = 0.0

        try:
            max_w = float(self.max_weight_input.text().strip())
        except ValueError:
            max_w = float(self.global_max_w)

        min_w = max(0.0, min(min_w, float(self.global_max_w)))
        max_w = max(min_w, min(max_w, float(self.global_max_w)))

        self.min_weight_input.setText(f"{min_w:.2f}")
        self.max_weight_input.setText(f"{max_w:.2f}")

        denom = self.global_max_w if self.global_max_w > 0 else 1.0
        pct_low = int(round((min_w / denom) * 100.0))
        pct_high = int(round((max_w / denom) * 100.0))

        self.range_slider.blockSignals(True)
        self.range_slider.setValues(pct_low, pct_high)
        self.range_slider.blockSignals(False)

        self.apply_thresholds(min_w, max_w)

    def on_left_interaction(self, obj, event):
        self.sync_clipping_ranges()
        if self.lock_camera_checkbox.isChecked():
            self.vtk_right.GetRenderWindow().Render()
        self.vtk_left.GetRenderWindow().Render()

    def on_right_interaction(self, obj, event):
        self.sync_clipping_ranges()
        if self.lock_camera_checkbox.isChecked():
            self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def open_matrix_window(self):
        self.update_matrix_view()
        self.matrix_dialog.show()

    def update_matrix_view(self):
        selected_list = sorted(list(self.selected_node_indices))

        weights_matrix_a = {}
        weights_matrix_b = {}

        for (u, v, wa) in getattr(self, 'active_edges_a', []):
            weights_matrix_a[(u, v)] = wa
            weights_matrix_a[(v, u)] = wa

        for (u, v, wb) in getattr(self, 'active_edges_b', []):
            weights_matrix_b[(u, v)] = wb
            weights_matrix_b[(v, u)] = wb

        for n in selected_list:
            weights_matrix_a[(n, n)] = self.node_weights_a[n]
            weights_matrix_b[(n, n)] = self.node_weights_b[n]

        self.matrix_dialog.draw_matrix(selected_list, weights_matrix_a, weights_matrix_b, 
                                        self.active_min_w, self.active_max_w)

    def _filter_highest_force_neighbor_edges(self, edge_list):
        if not edge_list:
            return []

        best_edge_per_node = {}

        for u, v, w in edge_list:
            if u not in best_edge_per_node or w > best_edge_per_node[u][2]:
                best_edge_per_node[u] = (u, v, w)

            if v not in best_edge_per_node or w > best_edge_per_node[v][2]:
                best_edge_per_node[v] = (u, v, w)

        unique_edges = {}
        for edge in best_edge_per_node.values():
            u, v, w = edge
            pair = (min(u, v), max(u, v))
            if pair not in unique_edges or w > unique_edges[pair][2]:
                unique_edges[pair] = (pair[0], pair[1], w)

        return list(unique_edges.values())

    def apply_thresholds(self, min_weight_threshold, max_weight_threshold):
        raw_edges_a = []
        raw_edges_b = []
        mode = self.edge_mode_combo.currentText()

        # 1. Edge-Level Thresholding per Dropdown Mode
        for (u, v, wa, wb) in self.all_edges:
            edge_a_active = (min_weight_threshold <= wa <= max_weight_threshold)
            edge_b_active = (min_weight_threshold <= wb <= max_weight_threshold)

            if mode == "Only from A":
                if edge_a_active:
                    raw_edges_a.append((u, v, wa))
            elif mode == "Only from B":
                if edge_b_active:
                    raw_edges_b.append((u, v, wb))
            elif mode == "Both (Intersection)":
                if edge_a_active and edge_b_active:
                    raw_edges_a.append((u, v, wa))
                    raw_edges_b.append((u, v, wb))
            else:  # Default (Union)
                if edge_a_active:
                    raw_edges_a.append((u, v, wa))
                if edge_b_active:
                    raw_edges_b.append((u, v, wb))

        # 2. Filter Highest Force Neighbor Chains if toggled
        if self.max_neighbor_toggle.isChecked():
            self.active_edges_a = self._filter_highest_force_neighbor_edges(raw_edges_a)
            self.active_edges_b = self._filter_highest_force_neighbor_edges(raw_edges_b)
        else:
            self.active_edges_a = raw_edges_a
            self.active_edges_b = raw_edges_b

        # 3. Node-Level Retention
        nodes_with_edges = set()
        for (u, v, _) in self.active_edges_a:
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)
        for (u, v, _) in self.active_edges_b:
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)

        if self.max_neighbor_toggle.isChecked():
            visible_nodes = set(nodes_with_edges)
        else:
            visible_nodes = set(nodes_with_edges)
            for i in range(self.num_nodes):
                if mode == "Only from A":
                    if min_weight_threshold <= self.node_weights_a[i] <= max_weight_threshold:
                        visible_nodes.add(i)
                elif mode == "Only from B":
                    if min_weight_threshold <= self.node_weights_b[i] <= max_weight_threshold:
                        visible_nodes.add(i)
                elif mode == "Both (Intersection)":
                    if (min_weight_threshold <= self.node_weights_a[i] <= max_weight_threshold) and \
                       (min_weight_threshold <= self.node_weights_b[i] <= max_weight_threshold):
                        visible_nodes.add(i)
                else:  # Default (Union)
                    if (min_weight_threshold <= self.node_weights_a[i] <= max_weight_threshold) or \
                       (min_weight_threshold <= self.node_weights_b[i] <= max_weight_threshold):
                        visible_nodes.add(i)

        self.active_visible_nodes_left = sorted(list(visible_nodes))

        if len(self.active_visible_nodes_left) == 0:
            self.left_actor_nodes.SetVisibility(False)
            self.left_actor_chains_a.SetVisibility(False)
            self.left_actor_chains_b.SetVisibility(False)
            self.sync_clipping_ranges()
            self.vtk_left.GetRenderWindow().Render()
            self.update_right_subgraph()
            return

        active_coords = self.coords[self.active_visible_nodes_left]
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(active_coords, deep=True))
        self.left_nodes_polydata.SetPoints(pts)

        self.update_particle_colors()
        self.left_actor_nodes.SetVisibility(True)

        delta_offset = 0.008

        # --- Volume A Parallel Lines ---
        lines_a = vtk.vtkCellArray()
        line_pts_a = vtk.vtkPoints()
        edge_weight_list_a = []
        pt_idx_a = 0

        for u, v, wa in self.active_edges_a:
            p1, p2 = self.coords[u], self.coords[v]
            vec = p2 - p1
            norm_vec = np.linalg.norm(vec)
            if norm_vec < 1e-9: continue
            u_dir = vec / norm_vec

            ref = np.array([0, 0, 1.0], dtype=np.float32) if abs(u_dir[2]) < 0.9 else np.array([0, 1.0, 0], dtype=np.float32)
            n_perp = np.cross(u_dir, ref)
            n_perp = n_perp / np.linalg.norm(n_perp)

            p1_offset = p1 - delta_offset * n_perp
            p2_offset = p2 - delta_offset * n_perp

            line_pts_a.InsertNextPoint(p1_offset)
            line_pts_a.InsertNextPoint(p2_offset)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pt_idx_a)
            line.GetPointIds().SetId(1, pt_idx_a + 1)
            lines_a.InsertNextCell(line)

            edge_weight_list_a.append(wa)
            pt_idx_a += 2

        # --- Volume B Parallel Lines ---
        lines_b = vtk.vtkCellArray()
        line_pts_b = vtk.vtkPoints()
        edge_weight_list_b = []
        pt_idx_b = 0

        for u, v, wb in self.active_edges_b:
            p1, p2 = self.coords[u], self.coords[v]
            vec = p2 - p1
            norm_vec = np.linalg.norm(vec)
            if norm_vec < 1e-9: continue
            u_dir = vec / norm_vec

            ref = np.array([0, 0, 1.0], dtype=np.float32) if abs(u_dir[2]) < 0.9 else np.array([0, 1.0, 0], dtype=np.float32)
            n_perp = np.cross(u_dir, ref)
            n_perp = n_perp / np.linalg.norm(n_perp)

            p1_offset = p1 + delta_offset * n_perp
            p2_offset = p2 + delta_offset * n_perp

            line_pts_b.InsertNextPoint(p1_offset)
            line_pts_b.InsertNextPoint(p2_offset)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pt_idx_b)
            line.GetPointIds().SetId(1, pt_idx_b + 1)
            lines_b.InsertNextCell(line)

            edge_weight_list_b.append(wb)
            pt_idx_b += 2

        line_width = 2.2 if self.max_neighbor_toggle.isChecked() else 1.0
        self.left_actor_chains_a.GetProperty().SetLineWidth(line_width)
        self.left_actor_chains_b.GetProperty().SetLineWidth(line_width)
        self.right_actor_chains_a.GetProperty().SetLineWidth(line_width)
        self.right_actor_chains_b.GetProperty().SetLineWidth(line_width)

        combined_active_weights = edge_weight_list_a + edge_weight_list_b
        if len(combined_active_weights) > 0:
            self.active_min_w = float(np.min(combined_active_weights))
            self.active_max_w = float(np.max(combined_active_weights))
            if self.active_min_w == self.active_max_w:
                self.active_max_w += 1e-5

            self.lut_a.SetTableRange(self.active_min_w, self.active_max_w)
            self.lut_b.SetTableRange(self.active_min_w, self.active_max_w)
            self.left_chains_a_mapper.SetScalarRange(self.active_min_w, self.active_max_w)
            self.left_chains_b_mapper.SetScalarRange(self.active_min_w, self.active_max_w)
            self.right_chains_a_mapper.SetScalarRange(self.active_min_w, self.active_max_w)
            self.right_chains_b_mapper.SetScalarRange(self.active_min_w, self.active_max_w)

        if len(edge_weight_list_a) > 0:
            arr_a = numpy_to_vtk(np.array(edge_weight_list_a, dtype=np.float32), deep=True)
            self.left_chains_a_polydata.SetPoints(line_pts_a)
            self.left_chains_a_polydata.SetLines(lines_a)
            self.left_chains_a_polydata.GetCellData().SetScalars(arr_a)
            self.left_actor_chains_a.SetVisibility(True)
        else:
            self.left_actor_chains_a.SetVisibility(False)

        if len(edge_weight_list_b) > 0:
            arr_b = numpy_to_vtk(np.array(edge_weight_list_b, dtype=np.float32), deep=True)
            self.left_chains_b_polydata.SetPoints(line_pts_b)
            self.left_chains_b_polydata.SetLines(lines_b)
            self.left_chains_b_polydata.GetCellData().SetScalars(arr_b)
            self.left_actor_chains_b.SetVisibility(True)
        else:
            self.left_actor_chains_b.SetVisibility(False)

        self.sync_clipping_ranges()
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
            real_ball_id = self.ball_ids[node_id]

            txt_source = vtk.vtkVectorText()
            txt_source.SetText(f"ID:{real_ball_id}")

            txt_mapper = vtk.vtkPolyDataMapper()
            txt_mapper.SetInputConnection(txt_source.GetOutputPort())

            txt_actor = vtk.vtkFollower()
            txt_actor.SetMapper(txt_mapper)
            txt_actor.SetCamera(active_cam)
            txt_actor.SetScale(0.04, 0.04, 0.04)
            txt_actor.SetPosition(pos[0] + 0.03, pos[1] + 0.03, pos[2] + 0.03)
            txt_actor.GetProperty().SetColor(1.0, 0.9, 0.2)
            txt_actor.GetProperty().SetLighting(False)

            self.ren_right.AddActor(txt_actor)
            self.text_label_actors_right.append(txt_actor)

    def update_right_subgraph(self):
        self.current_selected_nodes_right = sorted(list(self.selected_node_indices))

        if len(self.current_selected_nodes_right) == 0:
            self.right_actor_nodes.SetVisibility(False)
            self.right_actor_chains_a.SetVisibility(False)
            self.right_actor_chains_b.SetVisibility(False)
            self.update_right_particle_id_labels()
            self.sync_clipping_ranges()
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

        delta_offset = 0.008

        # --- Right Subgraph Volume A Parallel Edges ---
        lines_a = vtk.vtkCellArray()
        line_pts_a = vtk.vtkPoints()
        edge_weight_list_a = []
        pt_idx_a = 0

        for u, v, wa in self.active_edges_a:
            if u in self.selected_node_indices and v in self.selected_node_indices:
                p1, p2 = self.coords[u], self.coords[v]
                vec = p2 - p1
                norm_vec = np.linalg.norm(vec)
                if norm_vec < 1e-9: continue
                u_dir = vec / norm_vec

                ref = np.array([0, 0, 1.0], dtype=np.float32) if abs(u_dir[2]) < 0.9 else np.array([0, 1.0, 0], dtype=np.float32)
                n_perp = np.cross(u_dir, ref)
                n_perp = n_perp / np.linalg.norm(n_perp)

                p1_offset = p1 - delta_offset * n_perp
                p2_offset = p2 - delta_offset * n_perp

                line_pts_a.InsertNextPoint(p1_offset)
                line_pts_a.InsertNextPoint(p2_offset)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pt_idx_a)
                line.GetPointIds().SetId(1, pt_idx_a + 1)
                lines_a.InsertNextCell(line)

                edge_weight_list_a.append(wa)
                pt_idx_a += 2

        # --- Right Subgraph Volume B Parallel Edges ---
        lines_b = vtk.vtkCellArray()
        line_pts_b = vtk.vtkPoints()
        edge_weight_list_b = []
        pt_idx_b = 0

        for u, v, wb in self.active_edges_b:
            if u in self.selected_node_indices and v in self.selected_node_indices:
                p1, p2 = self.coords[u], self.coords[v]
                vec = p2 - p1
                norm_vec = np.linalg.norm(vec)
                if norm_vec < 1e-9: continue
                u_dir = vec / norm_vec

                ref = np.array([0, 0, 1.0], dtype=np.float32) if abs(u_dir[2]) < 0.9 else np.array([0, 1.0, 0], dtype=np.float32)
                n_perp = np.cross(u_dir, ref)
                n_perp = n_perp / np.linalg.norm(n_perp)

                p1_offset = p1 + delta_offset * n_perp
                p2_offset = p2 + delta_offset * n_perp

                line_pts_b.InsertNextPoint(p1_offset)
                line_pts_b.InsertNextPoint(p2_offset)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pt_idx_b)
                line.GetPointIds().SetId(1, pt_idx_b + 1)
                lines_b.InsertNextCell(line)

                edge_weight_list_b.append(wb)
                pt_idx_b += 2

        if len(edge_weight_list_a) > 0:
            arr_a = numpy_to_vtk(np.array(edge_weight_list_a, dtype=np.float32), deep=True)
            self.right_chains_a_polydata.SetPoints(line_pts_a)
            self.right_chains_a_polydata.SetLines(lines_a)
            self.right_chains_a_polydata.GetCellData().SetScalars(arr_a)
            self.right_actor_chains_a.SetVisibility(True)
        else:
            self.right_actor_chains_a.SetVisibility(False)

        if len(edge_weight_list_b) > 0:
            arr_b = numpy_to_vtk(np.array(edge_weight_list_b, dtype=np.float32), deep=True)
            self.right_chains_b_polydata.SetPoints(line_pts_b)
            self.right_chains_b_polydata.SetLines(lines_b)
            self.right_chains_b_polydata.GetCellData().SetScalars(arr_b)
            self.right_actor_chains_b.SetVisibility(True)
        else:
            self.right_actor_chains_b.SetVisibility(False)

        self.update_right_particle_id_labels()

        self.sync_clipping_ranges()
        self.vtk_right.GetRenderWindow().Render()

        if self.matrix_dialog.isVisible():
            self.update_matrix_view()

    def highlight_particle_center(self, point_idx):
        if self.hovered_node_idx == point_idx:
            return

        self.hovered_node_idx = point_idx
        self.update_particle_colors()

        self.sync_clipping_ranges()
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def clear_selection(self):
        self.selected_node_indices.clear()
        self.update_particle_colors()
        self.update_right_subgraph()
        self.sync_clipping_ranges()
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
            self.sync_clipping_ranges()
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
            self.sync_clipping_ranges()
            self.vtk_left.GetRenderWindow().Render()

        self.style_right.OnLeftButtonDown()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    
    base_font = app.font()
    base_font.setPointSize(9)
    app.setFont(base_font)

    viewer = DualPanelForceChainBuilder(initial_assembly=58, initial_timestep=80)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()