import os
import sys
import numpy as np
import pandas as pd

import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QCheckBox, QComboBox, QSplitter, QFrame, QDialog, QLineEdit, QSlider
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QDoubleValidator, QIntValidator, QFont
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Matplotlib integration for Matrix Rendering
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.colors as mcolors


# ==============================================================================
# EDGE ANISOTROPY COMPUTATION (Satake 1982 / Oda 1982)
# ==============================================================================
def compute_edge_anisotropy(edges, coords):
    """
    Computes purely geometric 3D edge orientation fabric anisotropy (a_edge)
    for an active set of contact edges without eigenvectors.
    """
    if not edges or len(edges) == 0:
        return {"count": 0, "a_edge": 0.0}

    branch_vectors = []
    for u, v, _ in edges:
        p1, p2 = coords[u], coords[v]
        vec = p2 - p1
        norm_v = np.linalg.norm(vec)
        if norm_v < 1e-9:
            continue
        branch_vectors.append(vec / norm_v)

    if not branch_vectors:
        return {"count": 0, "a_edge": 0.0}

    N_vec = np.array(branch_vectors, dtype=np.float64)
    M_count = len(N_vec)

    # Geometric Fabric Tensor: Phi = (1/M) sum(n_k * n_k^T)
    Phi = np.einsum('ki,kj->ij', N_vec, N_vec) / float(M_count)
    
    # Deviatoric invariant: a_edge = sqrt(3/2 * dev(Phi):dev(Phi))
    dev = Phi - (1.0 / 3.0) * np.eye(3)
    a_edge = np.sqrt(1.5 * np.sum(dev * dev))

    return {
        "count": M_count,
        "a_edge": float(np.clip(a_edge, 0.0, 1.0))
    }


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
        self.setFixedWidth(120)

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
        painter.setBrush(QBrush(QColor("#777777")))
        painter.drawRoundedRect(span_rect, 2.5, 2.5)

        # 3. Ruler Ticks Below Track
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
        painter.setPen(QColor("#ffffff"))

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
        for x in [low_x, high_x]:
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.setPen(QPen(QColor("#bbbbbb"), 1.2))
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
# CUSTOM SINGLE-HANDLE CHAIN LENGTH SLIDER WITH RULER TICKS & VALUE LABELS
# ==============================================================================
class ChainLengthSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, minimum=1, maximum=1, value=1, parent=None):
        super().__init__(parent)
        self.min_val = minimum
        self.max_val = maximum
        self.curr_val = value

        self.handle_radius = 6.5
        self.is_dragging = False

        self.setMinimumHeight(44)
        self.setFixedWidth(90)

    def setRange(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max(min_val, max_val)
        self.curr_val = max(self.min_val, min(self.curr_val, self.max_val))
        self.update()

    def setValue(self, val):
        val = max(self.min_val, min(val, self.max_val))
        if val != self.curr_val:
            self.curr_val = val
            self.update()
            self.valueChanged.emit(self.curr_val)

    def value(self):
        return self.curr_val

    def minimum(self):
        return self.min_val

    def maximum(self):
        return self.max_val

    def _val_to_pos(self, val):
        width = self.width() - 2 * self.handle_radius
        if self.max_val <= self.min_val:
            return self.handle_radius
        return self.handle_radius + (val - self.min_val) / (self.max_val - self.min_val) * width

    def _pos_to_val(self, pos):
        width = self.width() - 2 * self.handle_radius
        if width <= 0 or self.max_val <= self.min_val:
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

        painter.setBrush(QBrush(QColor("#333333")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 2.5, 2.5)

        h_x = self._val_to_pos(self.curr_val)
        span_rect = QRectF(self.handle_radius, cy - 2.5, max(0.0, h_x - self.handle_radius), 5)
        painter.setBrush(QBrush(QColor("#00ffaa") if self.isEnabled() else QColor("#555555")))
        painter.drawRoundedRect(span_rect, 2.5, 2.5)

        major_pen = QPen(QColor("#999999") if self.isEnabled() else QColor("#444444"), 1.0)
        minor_pen = QPen(QColor("#555555") if self.isEnabled() else QColor("#333333"), 0.8)

        tick_top = cy + 6.0
        pos_min = self._val_to_pos(self.min_val)
        pos_max = self._val_to_pos(self.max_val)
        mid_val = (self.min_val + self.max_val) // 2
        pos_mid = self._val_to_pos(mid_val)

        for i in range(1, 5):
            t_pct = i / 5.0
            tx = pos_min + t_pct * (pos_max - pos_min)
            painter.setPen(minor_pen)
            painter.drawLine(int(tx), int(tick_top), int(tx), int(tick_top + 3.0))

        painter.setPen(major_pen)
        painter.drawLine(int(pos_min), int(tick_top), int(pos_min), int(tick_top + 5.5))
        painter.drawLine(int(pos_mid), int(tick_top), int(pos_mid), int(tick_top + 4.5))
        painter.drawLine(int(pos_max), int(tick_top), int(pos_max), int(tick_top + 5.5))

        font = QFont()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#aaaaaa") if self.isEnabled() else QColor("#555555"))

        label_y = tick_top + 7.0
        label_h = 13.0

        rect_min = QRectF(pos_min - 15, label_y, 30, label_h)
        painter.drawText(rect_min, Qt.AlignHCenter | Qt.AlignTop, str(self.min_val))

        if self.max_val > self.min_val + 1:
            rect_mid = QRectF(pos_mid - 15, label_y, 30, label_h)
            painter.drawText(rect_mid, Qt.AlignHCenter | Qt.AlignTop, str(mid_val))

        if self.max_val > self.min_val:
            rect_max = QRectF(pos_max - 15, label_y, 30, label_h)
            painter.drawText(rect_max, Qt.AlignHCenter | Qt.AlignTop, str(self.max_val))

        handle_color = QColor("#00ffaa") if self.isEnabled() else QColor("#777777")
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(QRectF(h_x - self.handle_radius, cy - self.handle_radius, 
                                   2 * self.handle_radius, 2 * self.handle_radius))

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            val = self._pos_to_val(event.position().x())
            self.setValue(val)

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            return
        if self.is_dragging:
            val = self._pos_to_val(event.position().x())
            self.setValue(val)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False


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

    pred_npmncf_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_npmncf.csv")
    if not os.path.exists(pred_npmncf_path):
        pred_npmncf_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_predicted_npmncf.csv")

    for path in [pos_x_path, pos_y_path, pos_z_path, pred_npmncf_path]:
        if not os.path.exists(path):
            return None, None, None, None, None

    df_x = pd.read_csv(pos_x_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'x'])
    df_y = pd.read_csv(pos_y_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'y'])
    df_z = pd.read_csv(pos_z_path, sep=r'\s+', skiprows=2, header=None, names=['ball_id', 'z'])

    df_x['ball_id'] = df_x['ball_id'].astype(int)
    df_y['ball_id'] = df_y['ball_id'].astype(int)
    df_z['ball_id'] = df_z['ball_id'].astype(int)

    df_coords = df_x.merge(df_y, on='ball_id').merge(df_z, on='ball_id')

    df_pred = pd.read_csv(pred_npmncf_path)
    df_nodes = df_pred.merge(df_coords, on='ball_id', how='inner')

    ball_ids = df_nodes['ball_id'].values
    num_nodes = len(ball_ids)
    id_to_idx = {bid: idx for idx, bid in enumerate(ball_ids)}

    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float32) * 100.0

    if 'actual_npmncf' in df_nodes.columns:
        raw_node_weights_a = df_nodes['actual_npmncf'].values.astype(np.float32)
    else:
        fallback_col = [c for c in df_nodes.columns if 'actual' in c.lower()]
        raw_node_weights_a = df_nodes[fallback_col[0]].values.astype(np.float32) if fallback_col else np.ones(num_nodes, dtype=np.float32)

    raw_node_weights_b = df_nodes['predicted_npmncf'].values.astype(np.float32)

    mean_a = float(np.mean(raw_node_weights_a)) if len(raw_node_weights_a) > 0 else 0.0
    mean_b = float(np.mean(raw_node_weights_b)) if len(raw_node_weights_b) > 0 else 0.0
    print(f"[Timestep {timestep}] Volume A (Actual) Mean: {mean_a:.6f} | Volume B (Predicted) Mean: {mean_b:.6f}")

    end1_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end1.tab")
    end2_path = resolve_path(actual_dir, f"{assembly_id}_{timestep}_contact_end2.tab")
    pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_contact_force.csv")
    if not os.path.exists(pred_contact_path):
        pred_contact_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_predicted_contact_force.csv")
    contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_contact_pairs.csv")
    if not os.path.exists(contact_pairs_path):
        contact_pairs_path = resolve_path(predicted_dir, f"assembly_{assembly_id}_timestep_{timestep}_contact_pairs.csv")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    if os.path.exists(end1_path) and os.path.exists(end2_path):
        e1 = pd.read_csv(end1_path, sep=r'\s+', header=None, skiprows=2)
        e2 = pd.read_csv(end2_path, sep=r'\s+', header=None, skiprows=2)
        e1_ids = pd.to_numeric(e1.iloc[:, 1], errors='coerce').dropna().astype(int)
        e2_ids = pd.to_numeric(e2.iloc[:, 1], errors='coerce').dropna().astype(int)
        for u_id, v_id in zip(e1_ids, e2_ids):
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1

    target_pred_csv = pred_contact_path if os.path.exists(pred_contact_path) else (contact_pairs_path if os.path.exists(contact_pairs_path) else None)
    if target_pred_csv:
        df_contacts = pd.read_csv(target_pred_csv)
        for _, row in df_contacts.iterrows():
            u_id, v_id = int(row['sender_id']), int(row['receiver_id'])
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                adj_matrix[u, v] = adj_matrix[v, u] = 1

    return adj_matrix, raw_node_weights_a, raw_node_weights_b, coords, ball_ids


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
        self.setGeometry(30, 40, 1850, 880)

        adj_matrix, raw_node_weights_a, raw_node_weights_b, coords, ball_ids = load_real_and_predicted_data(
            self.current_assembly, self.current_timestep
        )

        self.adj_matrix = adj_matrix
        self.raw_node_weights_a = raw_node_weights_a
        self.raw_node_weights_b = raw_node_weights_b
        self.node_weights_a = raw_node_weights_a.copy()
        self.node_weights_b = raw_node_weights_b.copy()

        self.coords = coords
        self.ball_ids = ball_ids
        self.num_nodes = len(coords)
        self.id_to_idx = {bid: idx for idx, bid in enumerate(ball_ids)}

        self.compute_spatial_bounds()

        # Build initial graph edges
        self.build_graphs_from_arrays()

        all_nodes_max = max(float(np.max(self.node_weights_a)), float(np.max(self.node_weights_b)))
        self.global_min_w = 0.0
        self.global_max_w = all_nodes_max if all_nodes_max > 0 else 1.0
        self.active_min_w = 0.0
        self.active_max_w = self.global_max_w

        self.selected_node_indices = set()
        self.selected_nodes_a = set()
        self.selected_nodes_b = set()

        self.traced_chain_edges_a = set()
        self.traced_chain_edges_b = set()

        self.full_chain_paths_a = []
        self.full_chain_paths_b = []
        self.active_chain_root_node = -1
        self.hovered_node_idx = -1
        self.active_visible_nodes_left = []
        self.current_selected_nodes_right = []
        self.text_label_actors_right = []

        self.COLOR_NORMAL = (180, 180, 180)
        self.COLOR_HOVER = (0, 255, 255)
        self.COLOR_SELECTED = (255, 230, 0)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)

        white_input_style = """
            QLineEdit {
                font-weight: bold;
                font-size: 8.5pt;
                color: #ffffff;
                background-color: #222222;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 1px;
            }
        """

        btn_style = """
            QPushButton {
                font-weight: bold;
                font-size: 8.5pt;
                color: #ffffff;
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 2px;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
            QPushButton:disabled { color: #555555; border-color: #333333; }
        """

        # 1. Timestep Text Entry
        control_layout.addWidget(QLabel("<b>Timestep:</b>"))
        self.timestep_input = QLineEdit(str(self.current_timestep))
        self.timestep_input.setFixedWidth(36)
        self.timestep_input.setAlignment(Qt.AlignCenter)
        self.timestep_input.returnPressed.connect(self.handle_timestep_change)
        control_layout.addWidget(self.timestep_input)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(42)
        self.load_btn.setStyleSheet("font-weight: bold; background-color: #005588; color: white;")
        self.load_btn.clicked.connect(self.handle_timestep_change)
        control_layout.addWidget(self.load_btn)

        control_layout.addSpacing(3)

        # 2. Independent Normalization Toggles
        self.norm_a_checkbox = QCheckBox("Norm A")
        self.norm_a_checkbox.setChecked(False)
        self.norm_a_checkbox.setStyleSheet("font-weight: bold; color: #00e5ff; font-size: 8.5pt;")
        self.norm_a_checkbox.toggled.connect(self.on_normalization_toggled)
        control_layout.addWidget(self.norm_a_checkbox)

        self.norm_b_checkbox = QCheckBox("Norm B")
        self.norm_b_checkbox.setChecked(False)
        self.norm_b_checkbox.setStyleSheet("font-weight: bold; color: #ff9900; font-size: 8.5pt;")
        self.norm_b_checkbox.toggled.connect(self.on_normalization_toggled)
        control_layout.addWidget(self.norm_b_checkbox)

        control_layout.addSpacing(3)

        # 3. Dual Range Slider & Manual Text Input Bounds
        control_layout.addWidget(QLabel("<b>Range:</b>"))
        self.min_weight_input = QLineEdit("0.00")
        self.min_weight_input.setFixedWidth(38)
        self.min_weight_input.setAlignment(Qt.AlignCenter)
        self.min_weight_input.setStyleSheet(white_input_style)
        self.min_weight_input.returnPressed.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.min_weight_input)

        self.range_slider = RangeSlider(minimum=0, maximum=100, low=0, high=100)
        self.range_slider.setMaxForceVal(self.global_max_w)
        self.range_slider.valuesChanged.connect(self.on_range_changed)
        control_layout.addWidget(self.range_slider)

        self.max_weight_input = QLineEdit(f"{self.global_max_w:.2f}")
        self.max_weight_input.setFixedWidth(38)
        self.max_weight_input.setAlignment(Qt.AlignCenter)
        self.max_weight_input.setStyleSheet(white_input_style)
        self.max_weight_input.returnPressed.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.max_weight_input)

        self.apply_range_btn = QPushButton("Set")
        self.apply_range_btn.setFixedWidth(30)
        self.apply_range_btn.setStyleSheet("font-weight: bold; background-color: #444444; color: white; font-size: 8pt;")
        self.apply_range_btn.clicked.connect(self.on_manual_range_applied)
        control_layout.addWidget(self.apply_range_btn)

        self.update_range_validators()

        control_layout.addSpacing(3)

        # 4. Peak-Neighbor Chains Control
        self.max_neighbor_toggle = QCheckBox("Max-Force")
        self.max_neighbor_toggle.setChecked(False)
        self.max_neighbor_toggle.setStyleSheet("font-weight: bold; color: #ff55ff; font-size: 8.5pt;")
        self.max_neighbor_toggle.toggled.connect(self.on_max_neighbor_toggled)
        control_layout.addWidget(self.max_neighbor_toggle)

        control_layout.addSpacing(3)

        # 5. Slab Selection Dropdown & Controls
        control_layout.addWidget(QLabel("<b>Slab:</b>"))
        self.slab_mode_combo = QComboBox()
        self.slab_mode_combo.addItems(["Off", "Upper Slab", "Lower Slab"])
        self.slab_mode_combo.setStyleSheet("font-weight: bold; color: #ffeb3b; font-size: 8.5pt;")
        self.slab_mode_combo.currentIndexChanged.connect(self.on_slab_mode_changed)
        control_layout.addWidget(self.slab_mode_combo)

        self.slab_thick_label = QLabel("Thick: 5%")
        self.slab_thick_label.setStyleSheet("font-size: 8pt; color: #ffeb3b;")
        control_layout.addWidget(self.slab_thick_label)

        self.slab_thick_slider = QSlider(Qt.Horizontal)
        self.slab_thick_slider.setRange(1, 10)
        self.slab_thick_slider.setValue(5)
        self.slab_thick_slider.setFixedWidth(50)
        self.slab_thick_slider.setEnabled(False)
        self.slab_thick_slider.valueChanged.connect(self.on_slab_param_changed)
        control_layout.addWidget(self.slab_thick_slider)

        self.slab_k_val = 5
        self.slab_k_label = QLabel("K: 5")
        self.slab_k_label.setStyleSheet("font-size: 8pt; color: #ffeb3b; font-weight: bold;")
        control_layout.addWidget(self.slab_k_label)

        self.slab_k_minus_btn = QPushButton("-")
        self.slab_k_minus_btn.setFixedSize(18, 20)
        self.slab_k_minus_btn.setStyleSheet(btn_style)
        self.slab_k_minus_btn.setEnabled(False)
        self.slab_k_minus_btn.clicked.connect(self.on_slab_k_minus)
        control_layout.addWidget(self.slab_k_minus_btn)

        self.slab_k_plus_btn = QPushButton("+")
        self.slab_k_plus_btn.setFixedSize(18, 20)
        self.slab_k_plus_btn.setStyleSheet(btn_style)
        self.slab_k_plus_btn.setEnabled(False)
        self.slab_k_plus_btn.clicked.connect(self.on_slab_k_plus)
        control_layout.addWidget(self.slab_k_plus_btn)

        control_layout.addSpacing(3)

        # 6. Single Chain Origin Trace Toggle, Seed ID Entry, and Chain Length Controls
        self.trace_chain_toggle = QCheckBox("Trace Click")
        self.trace_chain_toggle.setChecked(False)
        self.trace_chain_toggle.setStyleSheet("font-weight: bold; color: #00ffaa; font-size: 8.5pt;")
        self.trace_chain_toggle.toggled.connect(self.on_trace_chain_toggled)
        control_layout.addWidget(self.trace_chain_toggle)

        self.seed_id_input = QLineEdit()
        self.seed_id_input.setFixedWidth(38)
        self.seed_id_input.setAlignment(Qt.AlignCenter)
        self.seed_id_input.setPlaceholderText("ID")
        self.seed_id_input.setEnabled(False)
        self.seed_id_input.setStyleSheet(white_input_style)
        self.seed_id_input.returnPressed.connect(self.on_seed_id_entered)
        control_layout.addWidget(self.seed_id_input)

        self.seed_apply_btn = QPushButton("Go")
        self.seed_apply_btn.setFixedWidth(26)
        self.seed_apply_btn.setEnabled(False)
        self.seed_apply_btn.setStyleSheet("font-weight: bold; background-color: #337755; color: white; font-size: 8pt;")
        self.seed_apply_btn.clicked.connect(self.on_seed_id_entered)
        control_layout.addWidget(self.seed_apply_btn)

        control_layout.addWidget(QLabel("<b>Len:</b>"))

        self.chain_len_minus_btn = QPushButton("-")
        self.chain_len_minus_btn.setFixedSize(18, 20)
        self.chain_len_minus_btn.setStyleSheet(btn_style)
        self.chain_len_minus_btn.setEnabled(False)
        self.chain_len_minus_btn.clicked.connect(self.on_chain_len_minus)
        control_layout.addWidget(self.chain_len_minus_btn)

        self.chain_length_slider = ChainLengthSlider(minimum=1, maximum=1, value=1)
        self.chain_length_slider.setEnabled(False)
        self.chain_length_slider.valueChanged.connect(self.on_chain_length_slider_changed)
        control_layout.addWidget(self.chain_length_slider)

        self.chain_len_plus_btn = QPushButton("+")
        self.chain_len_plus_btn.setFixedSize(18, 20)
        self.chain_len_plus_btn.setStyleSheet(btn_style)
        self.chain_len_plus_btn.setEnabled(False)
        self.chain_len_plus_btn.clicked.connect(self.on_chain_len_plus)
        control_layout.addWidget(self.chain_len_plus_btn)

        self.chain_length_input = QLineEdit("1")
        self.chain_length_input.setFixedWidth(28)
        self.chain_length_input.setAlignment(Qt.AlignCenter)
        self.chain_length_input.setEnabled(False)
        self.chain_length_input.setStyleSheet("""
            QLineEdit {
                font-weight: bold;
                font-size: 8.5pt;
                color: #00ffaa;
                background-color: #222222;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 1px;
            }
        """)
        self.chain_length_validator = QIntValidator(1, 1, self)
        self.chain_length_input.setValidator(self.chain_length_validator)
        self.chain_length_input.returnPressed.connect(self.on_chain_length_text_applied)
        control_layout.addWidget(self.chain_length_input)

        control_layout.addSpacing(3)

        # 7. Dedicated Gauss Map Toggle
        self.gauss_toggle = QCheckBox("Gauss Map (Left View)")
        self.gauss_toggle.setChecked(False)
        self.gauss_toggle.setEnabled(False)
        self.gauss_toggle.setToolTip("Select 'Only from A' or 'Only from B' in Edge Filter to enable Gauss Map")
        self.gauss_toggle.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #ff77ff; 
                font-size: 8.5pt;
            }
            QCheckBox:disabled {
                color: #666666;
            }
        """)
        self.gauss_toggle.toggled.connect(self.on_gauss_toggled)
        control_layout.addWidget(self.gauss_toggle)

        control_layout.addSpacing(3)

        # 8. Camera Sync & ID Toggles
        self.lock_camera_checkbox = QCheckBox("Lock Cam")
        self.lock_camera_checkbox.setChecked(True)
        self.lock_camera_checkbox.setStyleSheet("font-weight: bold; color: #00e6ff; font-size: 8.5pt;")
        self.lock_camera_checkbox.toggled.connect(self.on_camera_lock_toggled)
        control_layout.addWidget(self.lock_camera_checkbox)

        self.show_ids_checkbox = QCheckBox("IDs")
        self.show_ids_checkbox.setChecked(True)
        self.show_ids_checkbox.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 8.5pt;")
        self.show_ids_checkbox.toggled.connect(self.on_show_ids_toggled)
        control_layout.addWidget(self.show_ids_checkbox)

        control_layout.addSpacing(3)

        # 9. Edge Filter Dropdown
        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItems(["Default (Union)", "Only from A", "Only from B", "Both (Intersection)"])
        self.edge_mode_combo.setStyleSheet("font-weight: bold; padding: 2px 4px; font-size: 8.5pt;")
        self.edge_mode_combo.currentIndexChanged.connect(self.on_edge_mode_changed)
        control_layout.addWidget(self.edge_mode_combo)

        control_layout.addSpacing(3)

        # 10. Matrix & Clear Action Buttons
        self.matrix_btn = QPushButton("Matrix")
        self.matrix_btn.setStyleSheet("font-weight: bold; padding: 3px 6px; background-color: #005588; color: white; font-size: 8.5pt;")
        self.matrix_btn.clicked.connect(self.open_matrix_window)
        control_layout.addWidget(self.matrix_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("font-weight: bold; padding: 3px 6px; font-size: 8.5pt;")
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

        # Initialize sliders and default render
        self.range_slider.setValues(0, 100)
        self.ren_left.ResetCamera()
        self.sync_clipping_ranges()

    def build_graphs_from_arrays(self):
        self.all_edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i, j] == 1:
                    wa = min(self.node_weights_a[i], self.node_weights_a[j])
                    wb = min(self.node_weights_b[i], self.node_weights_b[j])
                    self.all_edges.append((i, j, wa, wb))

    def recalculate_active_weights(self):
        if self.norm_a_checkbox.isChecked():
            mean_a = float(np.mean(self.raw_node_weights_a)) if len(self.raw_node_weights_a) > 0 else 1.0
            self.node_weights_a = (self.raw_node_weights_a / mean_a).astype(np.float32) if mean_a > 0 else self.raw_node_weights_a.copy()
        else:
            self.node_weights_a = self.raw_node_weights_a.copy()

        if self.norm_b_checkbox.isChecked():
            mean_b = float(np.mean(self.raw_node_weights_b)) if len(self.raw_node_weights_b) > 0 else 1.0
            self.node_weights_b = (self.raw_node_weights_b / mean_b).astype(np.float32) if mean_b > 0 else self.raw_node_weights_b.copy()
        else:
            self.node_weights_b = self.raw_node_weights_b.copy()

        self.build_graphs_from_arrays()

        all_nodes_max = max(float(np.max(self.node_weights_a)), float(np.max(self.node_weights_b)))
        self.global_min_w = 0.0
        self.global_max_w = all_nodes_max if all_nodes_max > 0 else 1.0

        self.range_slider.setMaxForceVal(self.global_max_w)
        self.max_weight_input.setText(f"{self.global_max_w:.2f}")
        self.update_range_validators()

        self.reset_chain_length_to_one()
        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def on_normalization_toggled(self):
        self.recalculate_active_weights()

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
        self.volume_span = float(np.linalg.norm(max_c - min_c)) or 1.0
        self.z_min = float(min_c[2])
        self.z_max = float(max_c[2])

    def reset_chain_length_to_one(self):
        self.chain_length_slider.blockSignals(True)
        self.chain_length_input.blockSignals(True)
        self.chain_length_slider.setValue(1)
        self.chain_length_input.setText("1")
        self.chain_length_slider.blockSignals(False)
        self.chain_length_input.blockSignals(False)

    def on_slab_k_minus(self):
        if self.slab_k_val > 1:
            self.slab_k_val -= 1
            self.slab_k_label.setText(f"K: {self.slab_k_val}")
            if self.slab_mode_combo.currentText() != "Off":
                self.reset_chain_length_to_one()
                self._recalculate_slab_chains()

    def on_slab_k_plus(self):
        self.slab_k_val += 1
        self.slab_k_label.setText(f"K: {self.slab_k_val}")
        if self.slab_mode_combo.currentText() != "Off":
            self.reset_chain_length_to_one()
            self._recalculate_slab_chains()

    def on_chain_len_minus(self):
        curr = self.chain_length_slider.value()
        if curr > self.chain_length_slider.minimum():
            self.chain_length_slider.setValue(curr - 1)

    def on_chain_len_plus(self):
        curr = self.chain_length_slider.value()
        if curr < self.chain_length_slider.maximum():
            self.chain_length_slider.setValue(curr + 1)

    def on_gauss_toggled(self, checked):
        if checked:
            if self.trace_chain_toggle.isChecked():
                self.trace_chain_toggle.setChecked(False)
            if self.slab_mode_combo.currentText() != "Off":
                self.slab_mode_combo.setCurrentText("Off")

        self.update_right_subgraph()

    def on_slab_mode_changed(self, index):
        mode = self.slab_mode_combo.currentText()
        is_active = (mode != "Off")

        self.slab_thick_slider.setEnabled(is_active)
        self.slab_k_minus_btn.setEnabled(is_active)
        self.slab_k_plus_btn.setEnabled(is_active)
        self.chain_length_slider.setEnabled(is_active)
        self.chain_length_input.setEnabled(is_active)
        self.chain_len_minus_btn.setEnabled(is_active)
        self.chain_len_plus_btn.setEnabled(is_active)

        if is_active:
            if self.gauss_toggle.isChecked():
                self.gauss_toggle.setChecked(False)

            self.trace_chain_toggle.blockSignals(True)
            self.trace_chain_toggle.setChecked(False)
            self.trace_chain_toggle.blockSignals(False)

            self.seed_id_input.setEnabled(False)
            self.seed_apply_btn.setEnabled(False)
            self.seed_id_input.clear()

            self.reset_chain_length_to_one()
            self._recalculate_slab_chains()
        else:
            self.full_chain_paths_a.clear()
            self.full_chain_paths_b.clear()
            self.selected_nodes_a.clear()
            self.selected_nodes_b.clear()
            self.traced_chain_edges_a.clear()
            self.traced_chain_edges_b.clear()
            self.reset_chain_length_to_one()
            self.update_particle_colors()
            self.update_right_subgraph()
            self.sync_clipping_ranges()
            self.vtk_left.GetRenderWindow().Render()

    def on_slab_param_changed(self):
        pct = self.slab_thick_slider.value()
        self.slab_thick_label.setText(f"Thick: {pct}%")

        if self.slab_mode_combo.currentText() != "Off":
            self.reset_chain_length_to_one()
            self._recalculate_slab_chains()

    def _recalculate_slab_chains(self):
        mode = self.slab_mode_combo.currentText()
        if mode == "Off":
            return

        pct = self.slab_thick_slider.value()
        k = self.slab_k_val
        z_thickness = (pct / 100.0) * (self.z_max - self.z_min)

        if mode == "Upper Slab":
            z_threshold = self.z_max - z_thickness
            slab_candidates = [
                idx for idx in self.active_visible_nodes_left 
                if self.coords[idx, 2] >= z_threshold
            ]
        else:
            z_threshold = self.z_min + z_thickness
            slab_candidates = [
                idx for idx in self.active_visible_nodes_left 
                if self.coords[idx, 2] <= z_threshold
            ]

        if not slab_candidates:
            self.full_chain_paths_a.clear()
            self.full_chain_paths_b.clear()
            self.selected_nodes_a.clear()
            self.selected_nodes_b.clear()
            self.traced_chain_edges_a.clear()
            self.traced_chain_edges_b.clear()
            self.update_particle_colors()
            self.update_right_subgraph()
            self.sync_clipping_ranges()
            self.vtk_left.GetRenderWindow().Render()
            return

        slab_candidates.sort(key=lambda idx: max(self.node_weights_a[idx], self.node_weights_b[idx]), reverse=True)
        top_k_seeds = slab_candidates[:k]

        self.full_chain_paths_a = []
        self.full_chain_paths_b = []

        max_possible_len = 1
        for seed in top_k_seeds:
            path_a = self._trace_full_single_volume_path(seed, self.active_edges_a, self.node_weights_a)
            path_b = self._trace_full_single_volume_path(seed, self.active_edges_b, self.node_weights_b)
            self.full_chain_paths_a.append(path_a)
            self.full_chain_paths_b.append(path_b)
            max_possible_len = max(max_possible_len, len(path_a), len(path_b))

        self.chain_length_slider.blockSignals(True)
        self.chain_length_input.blockSignals(True)
        self.chain_length_slider.setRange(1, max_possible_len)
        self.chain_length_validator.setRange(1, max_possible_len)
        self.chain_length_slider.setValue(1)
        self.chain_length_input.setText("1")
        self.chain_length_slider.blockSignals(False)
        self.chain_length_input.blockSignals(False)

        self._apply_chain_length(1)

    def on_seed_id_entered(self):
        text = self.seed_id_input.text().strip()
        if not text:
            return
        try:
            bid = int(text)
        except ValueError:
            return

        if bid not in self.id_to_idx:
            return

        node_idx = self.id_to_idx[bid]

        if not self.trace_chain_toggle.isChecked():
            self.trace_chain_toggle.setChecked(True)

        self.reset_chain_length_to_one()
        self._recalculate_full_chains_and_update(node_idx)

    def handle_timestep_change(self):
        text = self.timestep_input.text().strip()
        try:
            new_timestep = int(text)
        except ValueError:
            return

        adj_matrix, raw_node_weights_a, raw_node_weights_b, coords, ball_ids = load_real_and_predicted_data(
            self.current_assembly, new_timestep
        )

        if coords is None:
            return

        self.current_timestep = new_timestep
        self.adj_matrix = adj_matrix
        self.raw_node_weights_a = raw_node_weights_a
        self.raw_node_weights_b = raw_node_weights_b
        self.coords = coords
        self.ball_ids = ball_ids
        self.num_nodes = len(coords)
        self.id_to_idx = {bid: idx for idx, bid in enumerate(ball_ids)}

        self.compute_spatial_bounds()
        self.setWindowTitle(f"Parallel Force Chains - (Assembly {self.current_assembly}, Timestep {self.current_timestep})")

        self.recalculate_active_weights()

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

    def create_lut_gauss(self):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            r = 1.0 - t
            g = 1.0 - 0.92 * t
            b = 1.0 - 0.45 * t
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

        # ======================================================================
        # 3D GAUSS MAP SPHERE & COLORBAR ACTORS
        # ======================================================================
        self.lut_gauss = self.create_lut_gauss()

        self.gauss_surface_polydata = vtk.vtkPolyData()
        self.gauss_surface_mapper = vtk.vtkPolyDataMapper()
        self.gauss_surface_mapper.SetInputData(self.gauss_surface_polydata)
        self.gauss_surface_mapper.SetLookupTable(self.lut_gauss)
        self.gauss_surface_mapper.SetScalarModeToUseCellData()

        self.gauss_surface_actor = vtk.vtkActor()
        self.gauss_surface_actor.SetMapper(self.gauss_surface_mapper)
        self.gauss_surface_actor.GetProperty().SetOpacity(1.0)
        self.gauss_surface_actor.GetProperty().SetSpecular(0.20)
        self.gauss_surface_actor.SetVisibility(False)
        self.ren_right.AddActor(self.gauss_surface_actor)

        self.gauss_wire_mapper = vtk.vtkPolyDataMapper()
        self.gauss_wire_mapper.SetInputData(self.gauss_surface_polydata)
        self.gauss_wire_mapper.ScalarVisibilityOff()

        self.gauss_wire_actor = vtk.vtkActor()
        self.gauss_wire_actor.SetMapper(self.gauss_wire_mapper)
        self.gauss_wire_actor.GetProperty().SetRepresentationToWireframe()
        self.gauss_wire_actor.GetProperty().SetColor(0.4, 0.45, 0.55)
        self.gauss_wire_actor.GetProperty().SetLineWidth(0.6)
        self.gauss_wire_actor.SetVisibility(False)
        self.ren_right.AddActor(self.gauss_wire_actor)

        # Integer Ticks Colorbar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut_gauss)
        self.scalar_bar.SetTitle("")
        self.scalar_bar.SetNumberOfLabels(0)
        self.scalar_bar.DrawAnnotationsOn()
        self.scalar_bar.SetAnnotationTextScaling(0)
        self.scalar_bar.SetWidth(0.10)
        self.scalar_bar.SetHeight(0.45)
        self.scalar_bar.GetPositionCoordinate().SetValue(0.87, 0.28)
        self.scalar_bar.GetAnnotationTextProperty().SetColor(0.95, 0.95, 0.95)
        self.scalar_bar.GetAnnotationTextProperty().SetFontSize(9)
        self.scalar_bar.GetAnnotationTextProperty().BoldOn()
        self.scalar_bar.GetAnnotationTextProperty().ShadowOff()
        self.scalar_bar.SetVisibility(False)
        self.ren_right.AddActor2D(self.scalar_bar)

        # ======================================================================
        # EDGE ANISOTROPY DISPLAY OVERLAY
        # ======================================================================
        self.anisotropy_text_actor = vtk.vtkTextActor()
        self.anisotropy_text_actor.GetTextProperty().SetFontSize(13)
        self.anisotropy_text_actor.GetTextProperty().SetColor(0.0, 1.0, 0.8)
        self.anisotropy_text_actor.GetTextProperty().BoldOn()
        self.anisotropy_text_actor.GetTextProperty().ShadowOn()
        self.anisotropy_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.anisotropy_text_actor.GetPositionCoordinate().SetValue(0.04, 0.88)
        self.anisotropy_text_actor.SetVisibility(False)
        self.ren_right.AddActor2D(self.anisotropy_text_actor)

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
        self.reset_chain_length_to_one()
        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def on_trace_chain_toggled(self, checked):
        if checked:
            if self.gauss_toggle.isChecked():
                self.gauss_toggle.setChecked(False)

            if self.slab_mode_combo.currentText() != "Off":
                self.slab_mode_combo.blockSignals(True)
                self.slab_mode_combo.setCurrentText("Off")
                self.slab_mode_combo.blockSignals(False)
                self.slab_thick_slider.setEnabled(False)
                self.slab_k_minus_btn.setEnabled(False)
                self.slab_k_plus_btn.setEnabled(False)

        self.chain_length_slider.setEnabled(checked)
        self.chain_length_input.setEnabled(checked)
        self.chain_len_minus_btn.setEnabled(checked)
        self.chain_len_plus_btn.setEnabled(checked)
        self.seed_id_input.setEnabled(checked)
        self.seed_apply_btn.setEnabled(checked)

        if not checked:
            self.selected_nodes_a.clear()
            self.selected_nodes_b.clear()
            self.traced_chain_edges_a.clear()
            self.traced_chain_edges_b.clear()
            self.full_chain_paths_a.clear()
            self.full_chain_paths_b.clear()
            self.active_chain_root_node = -1
            self.seed_id_input.clear()
            self.reset_chain_length_to_one()
            self.update_particle_colors()
            self.update_right_subgraph()
            self.sync_clipping_ranges()
            self.vtk_left.GetRenderWindow().Render()
        elif self.active_chain_root_node != -1:
            self.reset_chain_length_to_one()
            self._recalculate_full_chains_and_update(self.active_chain_root_node)

    def on_chain_length_slider_changed(self, val):
        self.chain_length_input.blockSignals(True)
        self.chain_length_input.setText(str(val))
        self.chain_length_input.blockSignals(False)

        is_slab_active = (self.slab_mode_combo.currentText() != "Off")
        if (self.trace_chain_toggle.isChecked() and self.active_chain_root_node != -1) or is_slab_active:
            self._apply_chain_length(val)

    def on_chain_length_text_applied(self):
        text = self.chain_length_input.text().strip()
        try:
            val = int(text)
        except ValueError:
            val = self.chain_length_slider.value()

        val = max(self.chain_length_slider.minimum(), min(val, self.chain_length_slider.maximum()))
        self.chain_length_slider.setValue(val)
        self.chain_length_input.setText(str(val))

        is_slab_active = (self.slab_mode_combo.currentText() != "Off")
        if (self.trace_chain_toggle.isChecked() and self.active_chain_root_node != -1) or is_slab_active:
            self._apply_chain_length(val)

    def _recalculate_full_chains_and_update(self, start_node):
        self.active_chain_root_node = start_node
        self.seed_id_input.blockSignals(True)
        self.seed_id_input.setText(str(self.ball_ids[start_node]))
        self.seed_id_input.blockSignals(False)

        path_a = self._trace_full_single_volume_path(start_node, self.active_edges_a, self.node_weights_a)
        path_b = self._trace_full_single_volume_path(start_node, self.active_edges_b, self.node_weights_b)

        self.full_chain_paths_a = [path_a]
        self.full_chain_paths_b = [path_b]

        max_possible_len = max(len(path_a), len(path_b), 1)

        self.chain_length_slider.blockSignals(True)
        self.chain_length_input.blockSignals(True)
        self.chain_length_slider.setRange(1, max_possible_len)
        self.chain_length_validator.setRange(1, max_possible_len)

        current_len = min(self.chain_length_slider.value(), max_possible_len)
        if current_len < 1:
            current_len = 1

        self.chain_length_slider.setValue(current_len)
        self.chain_length_input.setText(str(current_len))

        self.chain_length_slider.blockSignals(False)
        self.chain_length_input.blockSignals(False)

        self._apply_chain_length(current_len)

    def _apply_chain_length(self, target_length):
        nodes_a = set()
        nodes_b = set()
        edges_a = set()
        edges_b = set()

        for path_a in self.full_chain_paths_a:
            sub_a = path_a[:target_length]
            nodes_a.update(sub_a)
            for i in range(len(sub_a) - 1):
                edges_a.add((min(sub_a[i], sub_a[i+1]), max(sub_a[i], sub_a[i+1])))

        for path_b in self.full_chain_paths_b:
            sub_b = path_b[:target_length]
            nodes_b.update(sub_b)
            for i in range(len(sub_b) - 1):
                edges_b.add((min(sub_b[i], sub_b[i+1]), max(sub_b[i], sub_b[i+1])))

        self.selected_nodes_a = nodes_a
        self.selected_nodes_b = nodes_b
        self.selected_node_indices = self.selected_nodes_a.union(self.selected_nodes_b)

        self.traced_chain_edges_a = edges_a
        self.traced_chain_edges_b = edges_b

        self.update_particle_colors()
        self.update_right_subgraph()
        self.sync_clipping_ranges()
        self.vtk_left.GetRenderWindow().Render()

    def on_edge_mode_changed(self, index):
        mode = self.edge_mode_combo.currentText()
        is_single_volume = (mode in ["Only from A", "Only from B"])

        self.gauss_toggle.setEnabled(is_single_volume)
        if not is_single_volume and self.gauss_toggle.isChecked():
            self.gauss_toggle.setChecked(False)

        self.reset_chain_length_to_one()
        low, high = self.range_slider.getValues()
        self.on_range_changed(low, high)

    def on_range_changed(self, low_percentile, high_percentile):
        self.reset_chain_length_to_one()

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
        self.reset_chain_length_to_one()

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

    def _trace_full_single_volume_path(self, start_node, active_edges, node_weights):
        adj = {}
        for u, v, _ in active_edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        path = [start_node]
        visited = {start_node}
        current = start_node

        while True:
            neighbors = adj.get(current, set())
            unvisited = [nbr for nbr in neighbors if nbr not in visited]

            if not unvisited:
                break

            best_neighbor = max(unvisited, key=lambda nbr: node_weights[nbr])
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        return path

    def apply_thresholds(self, min_weight_threshold, max_weight_threshold):
        mode = self.edge_mode_combo.currentText()

        # ======================================================================
        # STEP 1: STRICT NODE FILTERING FIRST
        # ======================================================================
        nodes_a_pass = set(
            i for i in range(self.num_nodes)
            if min_weight_threshold <= self.node_weights_a[i] <= max_weight_threshold
        )
        nodes_b_pass = set(
            i for i in range(self.num_nodes)
            if min_weight_threshold <= self.node_weights_b[i] <= max_weight_threshold
        )

        if mode == "Only from A":
            active_node_set = nodes_a_pass
        elif mode == "Only from B":
            active_node_set = nodes_b_pass
        elif mode == "Both (Intersection)":
            active_node_set = nodes_a_pass.intersection(nodes_b_pass)
        else:
            active_node_set = nodes_a_pass.union(nodes_b_pass)

        self.active_visible_nodes_left = sorted(list(active_node_set))

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

        visible_set = set(self.active_visible_nodes_left)

        # ======================================================================
        # STEP 2: INDUCED EDGE FILTERING (BOTH ENDPOINTS MUST BE VISIBLE)
        # ======================================================================
        raw_edges_a = []
        raw_edges_b = []

        for (u, v, _, _) in self.all_edges:
            if u in visible_set and v in visible_set:
                w_edge_a = min(self.node_weights_a[u], self.node_weights_a[v])
                w_edge_b = min(self.node_weights_b[u], self.node_weights_b[v])

                if mode == "Only from A":
                    if u in nodes_a_pass and v in nodes_a_pass and w_edge_a > 0:
                        raw_edges_a.append((u, v, w_edge_a))
                elif mode == "Only from B":
                    if u in nodes_b_pass and v in nodes_b_pass and w_edge_b > 0:
                        raw_edges_b.append((u, v, w_edge_b))
                elif mode == "Both (Intersection)":
                    if (u in nodes_a_pass and v in nodes_a_pass and w_edge_a > 0) and \
                       (u in nodes_b_pass and v in nodes_b_pass and w_edge_b > 0):
                        raw_edges_a.append((u, v, w_edge_a))
                        raw_edges_b.append((u, v, w_edge_b))
                else:
                    if u in nodes_a_pass and v in nodes_a_pass and w_edge_a > 0:
                        raw_edges_a.append((u, v, w_edge_a))
                    if u in nodes_b_pass and v in nodes_b_pass and w_edge_b > 0:
                        raw_edges_b.append((u, v, w_edge_b))

        if self.max_neighbor_toggle.isChecked():
            self.active_edges_a = self._filter_highest_force_neighbor_edges(raw_edges_a)
            self.active_edges_b = self._filter_highest_force_neighbor_edges(raw_edges_b)
        else:
            self.active_edges_a = raw_edges_a
            self.active_edges_b = raw_edges_b

        # ======================================================================
        # STEP 3: BUILD VTK PARALLEL OFFSET LINES
        # ======================================================================
        delta_offset = 0.008

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

        self.active_min_w = float(min_weight_threshold)
        self.active_max_w = float(max_weight_threshold)
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

        if self.slab_mode_combo.currentText() != "Off":
            self._recalculate_slab_chains()
        elif self.trace_chain_toggle.isChecked() and self.active_chain_root_node != -1:
            self._recalculate_full_chains_and_update(self.active_chain_root_node)

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

        if not self.show_ids_checkbox.isChecked() or self.gauss_toggle.isChecked():
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

    def _generate_gauss_map_hemisphere(self, edges):
        R = self.volume_span * 0.4
        center = np.mean(self.coords, axis=0)

        phi_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        base_verts = [
            [-1,  phi_ratio, 0], [ 1,  phi_ratio, 0], [-1, -phi_ratio, 0], [ 1, -phi_ratio, 0],
            [0, -1,  phi_ratio], [0,  1,  phi_ratio], [0, -1, -phi_ratio], [0,  1, -phi_ratio],
            [ phi_ratio, 0, -1], [ phi_ratio, 0,  1], [-phi_ratio, 0, -1], [-phi_ratio, 0,  1]
        ]
        base_verts = [np.array(v, dtype=np.float32) / np.linalg.norm(v) for v in base_verts]

        base_faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        midpoint_cache = {}
        verts = list(base_verts)

        def get_midpoint(i1, i2):
            edge_key = (min(i1, i2), max(i1, i2))
            if edge_key in midpoint_cache:
                return midpoint_cache[edge_key]
            v_mid = (verts[i1] + verts[i2]) / 2.0
            v_mid = v_mid / np.linalg.norm(v_mid)
            idx = len(verts)
            verts.append(v_mid)
            midpoint_cache[edge_key] = idx
            return idx

        faces = base_faces
        for _ in range(2):
            new_faces = []
            for tri in faces:
                a, b, c = tri
                ab = get_midpoint(a, b)
                bc = get_midpoint(b, c)
                ca = get_midpoint(c, a)
                new_faces.extend([
                    [a, ab, ca],
                    [b, bc, ab],
                    [c, ca, bc],
                    [ab, bc, ca]
                ])
            faces = new_faces

        sphere_triangles = []
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            centroid = (v0 + v1 + v2) / 3.0
            c_unit = centroid / np.linalg.norm(centroid)

            normal = np.cross(v1 - v0, v2 - v0)
            if np.dot(normal, c_unit) < 0:
                v1, v2 = v2, v1

            sphere_triangles.append((v0, v1, v2, c_unit))

        N_bins = len(sphere_triangles)
        counts = np.zeros(N_bins, dtype=np.float32)

        centroids_mat = np.array([tri[3] for tri in sphere_triangles], dtype=np.float32)

        antipodal_indices = np.zeros(N_bins, dtype=int)
        for i in range(N_bins):
            antipodal_indices[i] = np.argmax(np.dot(centroids_mat, -centroids_mat[i]))

        for u, v, _ in edges:
            vec = self.coords[v] - self.coords[u]
            norm_v = np.linalg.norm(vec)
            if norm_v < 1e-9:
                continue
            n = vec / norm_v

            best_bin = np.argmax(np.dot(centroids_mat, n))
            opp_bin = antipodal_indices[best_bin]

            counts[best_bin] += 1.0
            counts[opp_bin] += 1.0

        max_count = float(np.max(counts))

        pts = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        cell_scalars = vtk.vtkFloatArray()
        cell_scalars.SetName("ContactCount")

        pt_counter = 0
        for i, (p0, p1, p2, _) in enumerate(sphere_triangles):
            pts.InsertNextPoint(center[0] + R * p0[0], center[1] + R * p0[1], center[2] + R * p0[2])
            pts.InsertNextPoint(center[0] + R * p1[0], center[1] + R * p1[1], center[2] + R * p1[2])
            pts.InsertNextPoint(center[0] + R * p2[0], center[1] + R * p2[1], center[2] + R * p2[2])

            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, pt_counter)
            tri.GetPointIds().SetId(1, pt_counter + 1)
            tri.GetPointIds().SetId(2, pt_counter + 2)
            polys.InsertNextCell(tri)

            cell_scalars.InsertNextValue(counts[i])
            pt_counter += 3

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetPolys(polys)
        polydata.GetCellData().SetScalars(cell_scalars)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOff()
        normals.ComputeCellNormalsOn()
        normals.Update()

        return normals.GetOutput(), 0.0, max(1.0, max_count)

    def update_right_subgraph(self):
        # MODE 1: 3D GAUSS MAP
        if self.gauss_toggle.isChecked():
            self.right_actor_nodes.SetVisibility(False)
            self.right_actor_chains_a.SetVisibility(False)
            self.right_actor_chains_b.SetVisibility(False)
            self.update_right_particle_id_labels()

            mode = self.edge_mode_combo.currentText()
            volume_label = "Volume A (Actual)" if mode == "Only from A" else "Volume B (Predicted)"
            edges = self.active_edges_a if mode == "Only from A" else self.active_edges_b
            polydata, min_c, max_c = self._generate_gauss_map_hemisphere(edges)

            # Compute Pure Edge Anisotropy
            metrics = compute_edge_anisotropy(edges, self.coords)

            if polydata.GetNumberOfPoints() > 0:
                max_int = int(round(max_c))
                max_c_clamped = max(1.0, float(max_int))

                self.gauss_surface_polydata.ShallowCopy(polydata)
                self.gauss_surface_mapper.SetScalarRange(0.0, max_c_clamped)
                self.lut_gauss.SetTableRange(0.0, max_c_clamped)

                # Generate integer ticks at step 10 ending with max_int
                self.lut_gauss.ResetAnnotations()
                tick_vals = list(range(0, max_int + 1, 10))
                if not tick_vals or tick_vals[-1] != max_int:
                    tick_vals.append(max_int)

                for tick in tick_vals:
                    self.lut_gauss.SetAnnotation(float(tick), str(tick))

                self.gauss_surface_actor.GetProperty().SetOpacity(1.0)
                self.gauss_surface_actor.SetVisibility(True)
                self.gauss_wire_actor.SetVisibility(True)
                self.scalar_bar.SetVisibility(True)

                # Concise HUD: Target, Edges, and Edge Anisotropy only
                hud_text = (
                    f"{volume_label}\n"
                    f"Active Edges: {metrics['count']}\n"
                    f"Edge Anisotropy: {metrics['a_edge']:.4f}"
                )
                self.anisotropy_text_actor.SetInput(hud_text)
                self.anisotropy_text_actor.SetVisibility(True)
            else:
                self.gauss_surface_actor.SetVisibility(False)
                self.gauss_wire_actor.SetVisibility(False)
                self.scalar_bar.SetVisibility(False)
                self.anisotropy_text_actor.SetVisibility(False)

            self.sync_clipping_ranges()
            self.vtk_right.GetRenderWindow().Render()
            return

        # MODE 2: STANDARD FORCE CHAINS & PARTICLES
        self.gauss_surface_actor.SetVisibility(False)
        self.gauss_wire_actor.SetVisibility(False)
        self.scalar_bar.SetVisibility(False)
        self.anisotropy_text_actor.SetVisibility(False)

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
        is_chain_active = self.trace_chain_toggle.isChecked() or (self.slab_mode_combo.currentText() != "Off")

        def is_edge_allowed_a(u, v):
            if is_chain_active:
                return (min(u, v), max(u, v)) in self.traced_chain_edges_a
            return (u in self.selected_node_indices and v in self.selected_node_indices)

        def is_edge_allowed_b(u, v):
            if is_chain_active:
                return (min(u, v), max(u, v)) in self.traced_chain_edges_b
            return (u in self.selected_node_indices and v in self.selected_node_indices)

        # Right Subgraph Volume A Parallel Edges
        lines_a = vtk.vtkCellArray()
        line_pts_a = vtk.vtkPoints()
        edge_weight_list_a = []
        pt_idx_a = 0

        for u, v, wa in self.active_edges_a:
            if is_edge_allowed_a(u, v):
                p1, p2 = self.coords[u].copy(), self.coords[v].copy()
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

        # Right Subgraph Volume B Parallel Edges
        lines_b = vtk.vtkCellArray()
        line_pts_b = vtk.vtkPoints()
        edge_weight_list_b = []
        pt_idx_b = 0

        for u, v, wb in self.active_edges_b:
            if is_edge_allowed_b(u, v):
                p1, p2 = self.coords[u].copy(), self.coords[v].copy()
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
        self.right_actor_chains_a.GetProperty().SetLineWidth(line_width)
        self.right_actor_chains_b.GetProperty().SetLineWidth(line_width)

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
        self.selected_nodes_a.clear()
        self.selected_nodes_b.clear()
        self.traced_chain_edges_a.clear()
        self.traced_chain_edges_b.clear()
        self.full_chain_paths_a.clear()
        self.full_chain_paths_b.clear()
        self.active_chain_root_node = -1
        self.seed_id_input.clear()
        self.reset_chain_length_to_one()
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
            if self.slab_mode_combo.currentText() != "Off" or self.gauss_toggle.isChecked():
                pass
            elif self.trace_chain_toggle.isChecked():
                self.reset_chain_length_to_one()
                self._recalculate_full_chains_and_update(picked_id)
            else:
                self.active_chain_root_node = -1
                self.seed_id_input.clear()
                self.selected_nodes_a.clear()
                self.selected_nodes_b.clear()
                self.traced_chain_edges_a.clear()
                self.traced_chain_edges_b.clear()
                self.full_chain_paths_a.clear()
                self.full_chain_paths_b.clear()
                self.reset_chain_length_to_one()
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
        if self.gauss_toggle.isChecked():
            self.style_right.OnMouseMove()
            return

        picked_id = self.get_picked_particle_id_right()
        self.highlight_particle_center(picked_id)
        self.style_right.OnMouseMove()

    def on_right_mouse_click(self, obj, event):
        if self.gauss_toggle.isChecked():
            self.style_right.OnLeftButtonDown()
            return

        picked_id = self.get_picked_particle_id_right()
        if picked_id != -1 and picked_id in self.selected_node_indices:
            self.selected_node_indices.remove(picked_id)
            self.selected_nodes_a.discard(picked_id)
            self.selected_nodes_b.discard(picked_id)
            self.traced_chain_edges_a.clear()
            self.traced_chain_edges_b.clear()
            self.full_chain_paths_a.clear()
            self.full_chain_paths_b.clear()
            self.active_chain_root_node = -1
            self.seed_id_input.clear()
            self.reset_chain_length_to_one()
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