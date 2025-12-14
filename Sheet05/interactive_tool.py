import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QRadioButton,
                            QFileDialog, QMessageBox, QButtonGroup, QGroupBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

from graphcut_core import GraphCutCore, compute_iou


class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid black;")
        self.setAlignment(Qt.AlignCenter)
        
        self.drawing = False
        self.last_point = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent_widget.image is not None:
            self.drawing = True
            self.parent_widget.save_to_history()
            pos = self.map_to_image(event.pos())
            if pos:
                self.last_point = pos
                self.parent_widget.draw_at(pos)
                
    def mouseMoveEvent(self, event):
        if self.drawing and self.parent_widget.image is not None:
            pos = self.map_to_image(event.pos())
            if pos:
                if self.last_point:
                    self.parent_widget.draw_line(self.last_point, pos)
                self.last_point = pos
                
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None
            
    def map_to_image(self, canvas_pos):
        if self.parent_widget.image is None:
            return None
            
        pixmap = self.pixmap()
        if pixmap is None:
            return None
            
        # Account for scaling and centering
        h, w = self.parent_widget.image.shape[:2]
        label_w, label_h = self.width(), self.height()
        
        scale = min(label_w / w, label_h / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        img_x = int((canvas_pos.x() - offset_x) / scale)
        img_y = int((canvas_pos.y() - offset_y) / scale)
        
        if 0 <= img_x < w and 0 <= img_y < h:
            return (img_x, img_y)
        return None


class InteractiveSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Graph Cut Segmentation")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data
        self.image = None  # Original RGB image (cv2 format - BGR)
        self.display_image = None  # Image with scribbles overlaid
        self.scribble_mask = None  # 0=unlabeled, 1=fg, 2=bg
        self.segmentation_mask = None
        self.ground_truth = None
        
        # Drawing state
        self.brush_size = 5
        self.current_label = 1  # 1=fg, 2=bg
        
        # History for undo
        self.history = []
        self.max_history = 20
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)
        
        # Load buttons
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet("background-color: lightblue; font-weight: bold; padding: 10px;")
        control_layout.addWidget(load_btn)
        
        load_gt_btn = QPushButton("Load Ground Truth (Optional)")
        load_gt_btn.clicked.connect(self.load_ground_truth)
        control_layout.addWidget(load_gt_btn)
        
        # Drawing mode
        mode_group = QGroupBox("Drawing Mode")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.fg_radio = QRadioButton("Foreground (White)")
        self.fg_radio.setChecked(True)
        self.fg_radio.toggled.connect(lambda: self.set_mode(1))
        self.mode_group.addButton(self.fg_radio)
        mode_layout.addWidget(self.fg_radio)
        
        self.bg_radio = QRadioButton("Background (Red)")
        self.bg_radio.toggled.connect(lambda: self.set_mode(2))
        self.mode_group.addButton(self.bg_radio)
        mode_layout.addWidget(self.bg_radio)
        
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)
        
        # Brush size
        brush_group = QGroupBox("Brush Size")
        brush_layout = QVBoxLayout()
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(5)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        self.brush_label = QLabel(f"Size: {self.brush_size}")
        brush_layout.addWidget(self.brush_label)
        brush_layout.addWidget(self.brush_slider)
        brush_group.setLayout(brush_layout)
        control_layout.addWidget(brush_group)
        
        # Model selection
        model_group = QGroupBox("Color Model")
        model_layout = QVBoxLayout()
        self.model_group = QButtonGroup()
        
        self.gmm_radio = QRadioButton("GMM")
        self.gmm_radio.setChecked(True)
        self.model_group.addButton(self.gmm_radio)
        model_layout.addWidget(self.gmm_radio)
        
        self.hist_radio = QRadioButton("Histogram")
        self.model_group.addButton(self.hist_radio)
        model_layout.addWidget(self.hist_radio)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        # GMM components
        gmm_group = QGroupBox("GMM Components")
        gmm_layout = QVBoxLayout()
        self.gmm_slider = QSlider(Qt.Horizontal)
        self.gmm_slider.setMinimum(2)
        self.gmm_slider.setMaximum(10)
        self.gmm_slider.setValue(5)
        self.gmm_label = QLabel("Components: 5")
        self.gmm_slider.valueChanged.connect(
            lambda v: self.gmm_label.setText(f"Components: {v}"))
        gmm_layout.addWidget(self.gmm_label)
        gmm_layout.addWidget(self.gmm_slider)
        gmm_group.setLayout(gmm_layout)
        control_layout.addWidget(gmm_group)
        
        # Histogram bins
        bins_group = QGroupBox("Histogram Bins")
        bins_layout = QVBoxLayout()
        self.bins_slider = QSlider(Qt.Horizontal)
        self.bins_slider.setMinimum(8)
        self.bins_slider.setMaximum(64)
        self.bins_slider.setValue(16)
        self.bins_label = QLabel("Bins: 16")
        self.bins_slider.valueChanged.connect(
            lambda v: self.bins_label.setText(f"Bins: {v}"))
        bins_layout.addWidget(self.bins_label)
        bins_layout.addWidget(self.bins_slider)
        bins_group.setLayout(bins_layout)
        control_layout.addWidget(bins_group)
        
        # Beta parameter
        beta_group = QGroupBox("Beta (×0.001)")
        beta_layout = QVBoxLayout()
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setMinimum(1)
        self.beta_slider.setMaximum(500)
        self.beta_slider.setValue(5)
        self.beta_label = QLabel("β: 0.005")
        self.beta_slider.valueChanged.connect(
            lambda v: self.beta_label.setText(f"β: {v/1000:.3f}"))
        beta_layout.addWidget(self.beta_label)
        beta_layout.addWidget(self.beta_slider)
        beta_group.setLayout(beta_layout)
        control_layout.addWidget(beta_group)
        
        # Lambda parameter
        lambda_group = QGroupBox("Lambda (Smoothness)")
        lambda_layout = QVBoxLayout()
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setMinimum(10)
        self.lambda_slider.setMaximum(300)
        self.lambda_slider.setValue(100)
        self.lambda_label = QLabel("λ: 100")
        self.lambda_slider.valueChanged.connect(
            lambda v: self.lambda_label.setText(f"λ: {v}"))
        lambda_layout.addWidget(self.lambda_label)
        lambda_layout.addWidget(self.lambda_slider)
        lambda_group.setLayout(lambda_layout)
        control_layout.addWidget(lambda_group)
        
        # Action buttons
        segment_btn = QPushButton("Run Segmentation\n(Space)")
        segment_btn.clicked.connect(self.run_segmentation)
        segment_btn.setStyleSheet("background-color: lightgreen; font-weight: bold; padding: 15px;")
        control_layout.addWidget(segment_btn)
        
        undo_btn = QPushButton("Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo)
        undo_btn.setStyleSheet("background-color: lightyellow; padding: 10px;")
        control_layout.addWidget(undo_btn)
        
        reset_btn = QPushButton("Reset All (R)")
        reset_btn.clicked.connect(self.reset)
        reset_btn.setStyleSheet("background-color: orange; padding: 10px;")
        control_layout.addWidget(reset_btn)
        
        save_btn = QPushButton("Save Mask (S)")
        save_btn.clicked.connect(self.save_mask)
        save_btn.setStyleSheet("background-color: lightcoral; padding: 10px;")
        control_layout.addWidget(save_btn)
        
        # IoU display
        self.iou_label = QLabel("IoU: N/A")
        self.iou_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: white; border: 2px solid black;")
        self.iou_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.iou_label)
        
        control_layout.addStretch()
        
        # Right panel - Canvas
        self.canvas = ImageCanvas(self)
        
        # Add to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas, stretch=1)
        
        # Keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self.run_segmentation)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("R"), self, self.reset)
        QShortcut(QKeySequence("S"), self, self.save_mask)
        
    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not filepath:
            return
            
        self.image = cv2.imread(filepath)  # Keep in BGR format
        
        h, w = self.image.shape[:2]
        self.scribble_mask = np.zeros((h, w), dtype=np.uint8)
        self.segmentation_mask = None
        self.history = []
        self.display_image = self.image.copy()
        
        self.update_canvas()
        
    def load_ground_truth(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Truth", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not filepath:
            return
            
        self.ground_truth = cv2.imread(filepath, 0)
        self.ground_truth = (self.ground_truth > 0).astype(np.uint8)
        QMessageBox.information(self, "Success", "Ground truth loaded!")
        
    def set_mode(self, label):
        self.current_label = label
        
    def update_brush_size(self, value):
        self.brush_size = value
        self.brush_label.setText(f"Size: {value}")
        
    def save_to_history(self):
        if self.scribble_mask is not None:
            self.history.append(self.scribble_mask.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
    def undo(self):
        if len(self.history) > 0:
            self.scribble_mask = self.history.pop()
            self.update_display()
            self.update_canvas()
        else:
            QMessageBox.warning(self, "Undo", "No more undo history!")
            
    def reset(self):
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.scribble_mask = np.zeros((h, w), dtype=np.uint8)
            self.segmentation_mask = None
            self.history = []
            self.display_image = self.image.copy()
            self.update_canvas()
            self.iou_label.setText("IoU: N/A")
            
    def draw_at(self, pos):
        x, y = pos
        cv2.circle(self.scribble_mask, (x, y), self.brush_size, self.current_label, -1)
        self.update_display()
        self.update_canvas()
        
    def draw_line(self, p1, p2):
        cv2.line(self.scribble_mask, p1, p2, self.current_label, self.brush_size * 2)
        self.update_display()
        self.update_canvas()
        
    def update_display(self):
        self.display_image = self.image.copy()
        
        # Draw foreground scribbles in white
        fg_mask = self.scribble_mask == 1
        self.display_image[fg_mask] = [255, 255, 255]  # BGR white
        
        # Draw background scribbles in red
        bg_mask = self.scribble_mask == 2
        self.display_image[bg_mask] = [0, 0, 255]  # BGR red
        
        # Overlay segmentation if available
        if self.segmentation_mask is not None:
            overlay = self.image.copy()
            overlay[self.segmentation_mask == 1] = [0, 255, 0]  # BGR green for FG
            self.display_image = cv2.addWeighted(self.display_image, 0.7, overlay, 0.3, 0)
            
    def update_canvas(self):
        if self.display_image is not None:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, 
                            QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit canvas
            scaled_pixmap = pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, 
                                         Qt.SmoothTransformation)
            self.canvas.setPixmap(scaled_pixmap)
            
    def run_segmentation(self):
        if self.image is None:
            QMessageBox.warning(self, "Error", "Please load an image first!")
            return
            
        if np.sum(self.scribble_mask == 1) == 0 or np.sum(self.scribble_mask == 2) == 0:
            QMessageBox.warning(self, "Error", 
                              "Please draw both foreground and background scribbles!")
            return
            
        try:
            # Get parameters
            use_gmm = self.gmm_radio.isChecked()
            beta = self.beta_slider.value() / 1000.0
            lam = self.lambda_slider.value()
            n_components = self.gmm_slider.value()
            bins = self.bins_slider.value()
            
            # Create GraphCutCore instance
            gc = GraphCutCore(self.image, self.scribble_mask)
            
            # Build color models
            if use_gmm:
                gc.build_color_models_gmm(n_components=n_components)
            else:
                gc.build_color_models_histogram(bins=bins)
            
            # Run graph cut
            self.segmentation_mask = gc.graph_cut(use_gmm=use_gmm, beta=beta, lam=lam)
            
            # Update display
            self.update_display()
            self.update_canvas()
            
            # Compute IoU if GT available
            if self.ground_truth is not None:
                iou = compute_iou(self.segmentation_mask, self.ground_truth)
                self.iou_label.setText(f"IoU: {iou:.4f}")
            else:
                self.iou_label.setText("IoU: N/A (no GT)")
                
            QMessageBox.information(self, "Success", "Segmentation completed!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def save_mask(self):
        if self.segmentation_mask is None:
            QMessageBox.warning(self, "Error", "No segmentation to save!")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "", "PNG Files (*.png)")
        if filepath:
            cv2.imwrite(filepath, self.segmentation_mask * 255)
            QMessageBox.information(self, "Success", f"Mask saved to:\n{filepath}")


def main():
    app = QApplication(sys.argv)
    window = InteractiveSegmentationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()