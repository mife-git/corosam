# -*- coding: utf-8 -*-
import sys
import time
import os
import numpy as np
import torch
import cv2
from skimage import io
from models.medsam.medsam import MedSAM
import pydicom
from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QColor,
    QImage,
    QCursor,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
    QComboBox,
    QLabel,
    QCheckBox,
    QGroupBox,
    QGridLayout,
    QSlider,
    QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer

# Constants
IMG_INPUT_SIZE = 256
CHECKPOINT = '../checkpoints/CoroSAM/CoroSAMSAM_Final_Training.pt'
DEVICE = "cpu"
POINT_RADIUS = 3
POINT_COLORS = {
    "Endpoint": QColor("red"),
    "Intermediate": QColor("blue"),
    "Eraser": QColor("green")
}
MASK_COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (0, 128, 0),  # Dark Green
]


def setup_model():
    """Initialize and return the segmentation model"""
    print("Loading model...")
    tic = time.perf_counter()

    net = MedSAM(in_chans=3, use_adapter=True, use_conv_adapter=True, channel_reduction=0.25)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu")['model'])
    net.eval()

    print(f"Model loaded in {time.perf_counter() - tic:.2f} seconds")
    return net


def create_point_channels(tips, branch_points, image_size):
    """
    Create two separate channels for points:
    - First channel: tip points with 4x4 squares
    - Second channel: branch points with 4x4 squares
    Args:
        tips: List of (x, y) coordinates for tip points
        branch_points: List of (x, y) coordinates for branch points
        image_size: Size of the input image (assumed square)

    Returns:
        tips_channel, branch_channel: Two numpy arrays containing point information
    """
    tips_channel = np.zeros((image_size, image_size), dtype=np.float32)
    branch_channel = np.zeros((image_size, image_size), dtype=np.float32)

    # Add tips to tips channel
    for point in tips:
        x, y = int(point[0]), int(point[1])
        y_min, y_max = max(0, y - 2), min(image_size, y + 2)
        x_min, x_max = max(0, x - 2), min(image_size, x + 2)
        tips_channel[y_min:y_max, x_min:x_max] = 1.0

    # Add branch points to branch channel
    for point in branch_points:
        x, y = int(point[0]), int(point[1])
        y_min, y_max = max(0, y - 2), min(image_size, y + 2)
        x_min, x_max = max(0, x - 2), min(image_size, x + 2)
        branch_channel[y_min:y_max, x_min:x_max] = 1.0

    return tips_channel, branch_channel


def np2pixmap(np_img):
    """Convert numpy array to QPixmap for display"""
    if np_img.dtype != np.uint8:
        np_img = (np_img * 255).astype(np.uint8)

    height, width, channel = np_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def create_circle_cursor(size):
    """Create a circular cursor with the specified size"""
    # Create an empty pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    # Create a painter to draw on the pixmap
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Draw a white circle with light gray border for better visibility
    painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
    painter.setBrush(QBrush(QColor(255, 255, 255, 128), Qt.SolidPattern))  # Semi-transparent white
    painter.drawEllipse(1, 1, size - 2, size - 2)

    # Draw a small crosshair at the center for precise positioning
    painter.setPen(QPen(Qt.black, 1))
    painter.drawLine(size // 2, size // 4, size // 2, 3 * size // 4)
    painter.drawLine(size // 4, size // 2, 3 * size // 4, size // 2)

    painter.end()

    # Create a cursor from the pixmap with the hot spot at the center
    return QCursor(pixmap, size // 2, size // 2)


# Load model at startup
model = setup_model()


class CustomGraphicsScene(QGraphicsScene):
    """Custom graphics scene that handles mouse events for the eraser tool"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setSceneRect(0, 0, 1000, 1000)  # Initial size
        self.mouse_pressed = False
        self.last_pos = None

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.mouse_pressed = True
        self.last_pos = event.scenePos()

        # Pass the event to the parent window
        if hasattr(self.parent_window, 'handle_mouse_press'):
            self.parent_window.handle_mouse_press(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events, used for continuous erasing"""
        if self.mouse_pressed and hasattr(self.parent_window, 'handle_mouse_move'):
            # Get the current position
            curr_pos = event.scenePos()

            # Pass both current and last position for smoother erasing
            self.parent_window.handle_mouse_move(self.last_pos, curr_pos)

            # Update last position
            self.last_pos = curr_pos

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        self.mouse_pressed = False
        if hasattr(self.parent_window, 'handle_mouse_release'):
            self.parent_window.handle_mouse_release(event)


class CustomGraphicsView(QGraphicsView):
    """Custom graphics view with zoom and pan capabilities"""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QGraphicsView.NoFrame)

        # Zoom parameters
        self.zoom_factor_base = 1.1  # Zoom factor per mouse wheel delta
        self.zoom = 1.0  # Current zoom level
        self.min_zoom = 0.1  # Minimum zoom level
        self.max_zoom = 10.0  # Maximum zoom level

        # For panning
        self.setDragMode(QGraphicsView.NoDrag)

    def wheelEvent(self, event):
        """Handle wheel events for zooming"""
        if event.angleDelta().y() > 0:
            zoom_factor = self.zoom_factor_base
        else:
            zoom_factor = 1 / self.zoom_factor_base

        new_zoom = self.zoom * zoom_factor
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom = new_zoom
            self.scale(zoom_factor, zoom_factor)

        # Prevent default behavior
        event.accept()

    def reset_zoom(self):
        """Reset view to original scale and position"""
        self.resetTransform()
        self.zoom = 1.0

    def toggle_drag_mode(self):
        """Toggle between pan and point-adding modes"""
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
            return False
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            return True


class SegmentationWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.advance_frame)
        self.playback_speed = 10  # Default 10 fps (frames per second)

        self.history = []  # action history
        self.max_history = 5  # max number of actions in the history
        self.current_state_index = -1

        # Initialize variables and ui blocks
        self.view = None
        self.scene = None
        self.bg_img = None

        # Image and processing data
        self.image_path = None
        self.original_image = None
        self.input_image = None
        self.input_h, self.input_w = IMG_INPUT_SIZE, IMG_INPUT_SIZE
        self.model_prediction = None
        self.binary_mask = None  # Pure binary mask from model
        self.current_mask = None  # Visualization mask
        self.eraser_size = 10  # Default eraser size
        self.is_erasing = False  # Flag to track eraser mode

        # DICOM specific variables
        self.dicom_dataset = None
        self.current_frame = 0
        self.total_frames = 0

        # Personalized cursors
        self.eraser_cursors = {}

        # Point data
        self.tips = []
        self.branch_points = []
        self.point_items = []  # For UI display
        self.curr_point_type = "Endpoint"  # Default point type
        self.mask_opacity = 0.4
        self.mask_color_idx = 0

        # Model inputs
        self.tips_channel = None
        self.branch_channel = None
        self.input_image_stack = None
        self.input_image_stack_tensor = None

        # Initialize UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Create custom scene for better mouse event handling
        self.scene = CustomGraphicsScene(self)

        # Use our custom graphics view for the image with zoom capability
        self.view = CustomGraphicsView(self.scene)
        main_layout.addWidget(self.view)

        # Controls section
        controls_layout = QGridLayout()

        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout()

        # Load image
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        load_button.setToolTip("Load an image for segmentation")

        # Load dicom
        load_dicom_button = QPushButton("Load DICOM")
        load_dicom_button.clicked.connect(self.load_dicom)
        load_dicom_button.setToolTip("Load DICOM file for segmentation")

        # Save
        save_button = QPushButton("Save Mask")
        save_button.clicked.connect(self.save_mask)
        save_button.setToolTip("Save the current segmentation mask")

        file_layout.addWidget(load_button)
        file_layout.addWidget(load_dicom_button)
        file_layout.addWidget(save_button)
        file_group.setLayout(file_layout)

        # Point controls group
        point_group = QGroupBox("Point Controls")
        point_layout = QGridLayout()

        # Point type selection
        """
        point_type_label = QLabel("Point Type:")
        self.point_type_combo = QComboBox()
        self.point_type_combo.addItems(["Endpoint", "Intermediate", "Eraser"])
        self.point_type_combo.currentTextChanged.connect(self.change_point_type)
        self.point_type_combo.setToolTip("Select the type of point to add")
        """

        self.eraser_mode_check = QCheckBox("Eraser Mode")
        self.eraser_mode_check.stateChanged.connect(self.toggle_eraser_mode)
        self.eraser_mode_check.setToolTip("Enable eraser mode (E key)")

        clear_points_button = QPushButton("Clear Points")
        clear_points_button.clicked.connect(self.clear_points)
        clear_points_button.setToolTip("Clear all annotation points")

        # Add Clear Mask button
        clear_mask_button = QPushButton("Clear Mask")
        clear_mask_button.clicked.connect(self.clear_mask)
        clear_mask_button.setToolTip("Clear the current segmentation mask")

        run_inference_button = QPushButton("Run Inference")
        run_inference_button.clicked.connect(self.run_inference)
        run_inference_button.setToolTip("Run the segmentation model")

        # Auto-inference checkbox
        self.auto_inference_check = QCheckBox("Auto Inference")
        self.auto_inference_check.setChecked(True)
        self.auto_inference_check.setToolTip("Automatically run inference when points are added")

        # Eraser size slider (mostra solo in modalità eraser)
        self.eraser_size_label = QLabel("Eraser Size:")
        self.eraser_size_slider = QSlider(Qt.Horizontal)
        self.eraser_size_slider.setMinimum(1)
        self.eraser_size_slider.setMaximum(20)
        self.eraser_size_slider.setValue(self.eraser_size)
        self.eraser_size_slider.setTickPosition(QSlider.TicksBelow)
        self.eraser_size_slider.setTickInterval(1)
        self.eraser_size_slider.valueChanged.connect(self.change_eraser_size)
        self.eraser_size_slider.setToolTip("Adjust the size of the eraser tool")

        # Inizialmente nascondi i controlli eraser
        self.eraser_size_label.setVisible(False)
        self.eraser_size_slider.setVisible(False)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_action)
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Undo last action (Ctrl+Z)")

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_action)
        self.redo_button.setEnabled(False)
        self.redo_button.setToolTip("Redo last undone action (Ctrl+Y)")

        point_layout.addWidget(self.eraser_mode_check, 3, 0)
        point_layout.addWidget(self.auto_inference_check, 3, 1)
        point_layout.addWidget(clear_points_button, 1, 0)
        point_layout.addWidget(clear_mask_button, 1, 1)
        point_layout.addWidget(run_inference_button, 1, 2)
        point_layout.addWidget(self.eraser_size_label, 3, 0)
        point_layout.addWidget(self.eraser_size_slider, 3, 1, 1, 2)
        point_layout.addWidget(self.undo_button, 2, 0)
        point_layout.addWidget(self.redo_button, 2, 1)
        point_group.setLayout(point_layout)

        mouse_info_group = QGroupBox("Mouse Controls")
        mouse_info_layout = QVBoxLayout()

        mouse_left_label = QLabel("Left Click: Add Endpoint (Green)")
        mouse_right_label = QLabel("Right Click: Add Branch Point (Blue)")
        mouse_eraser_label = QLabel("Eraser Mode + Left Drag: Erase mask")

        mouse_left_label.setStyleSheet("color: green; font-weight: bold;")
        mouse_right_label.setStyleSheet("color: blue; font-weight: bold;")
        mouse_eraser_label.setStyleSheet("color: red; font-weight: bold;")

        mouse_info_layout.addWidget(mouse_left_label)
        mouse_info_layout.addWidget(mouse_right_label)
        mouse_info_layout.addWidget(mouse_eraser_label)
        mouse_info_group.setLayout(mouse_info_layout)

        # DICOM controls
        dicom_group = QGroupBox("DICOM Controls")
        dicom_layout = QHBoxLayout()

        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_spinbox.valueChanged.connect(lambda value: self.set_frame(value))
        self.frame_spinbox.setToolTip("Select frame in DICOM series")

        self.frame_info_label = QLabel("No DICOM loaded")

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play/Pause automatic frame advancement (Space)")

        # Add the new control to the layout
        dicom_layout.addWidget(self.play_button)

        # Add a shortcut for play/pause
        self.shortcut_play = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play.activated.connect(lambda: self.play_button.click())

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.advance_frame)
        self.playback_fps = 10  # Fixed at 10 fps (frames per second)

        dicom_layout.addWidget(self.frame_spinbox)
        dicom_layout.addWidget(self.frame_info_label)
        dicom_group.setLayout(dicom_layout)

        # View controls group (for zoom and pan)
        view_group = QGroupBox("View Controls")
        view_layout = QHBoxLayout()

        reset_zoom_button = QPushButton("Reset Zoom")
        reset_zoom_button.clicked.connect(self.view.reset_zoom)
        reset_zoom_button.setToolTip("Reset zoom to original size (Z)")

        self.toggle_pan_button = QPushButton("Pan Mode")
        self.toggle_pan_button.setCheckable(True)
        self.toggle_pan_button.clicked.connect(self.toggle_pan_mode)
        self.toggle_pan_button.setToolTip("Toggle between pan and draw modes (P)")

        # Zoom labels
        zoom_in_label = QLabel("Zoom: Mouse wheel")
        zoom_in_label.setAlignment(Qt.AlignCenter)

        view_layout.addWidget(reset_zoom_button)
        view_layout.addWidget(self.toggle_pan_button)
        view_layout.addWidget(zoom_in_label)
        view_group.setLayout(view_layout)

        # Mask appearance group
        appearance_group = QGroupBox("Mask Appearance")
        appearance_layout = QHBoxLayout()

        opacity_label = QLabel("Opacity:")
        self.opacity_combo = QComboBox()
        self.opacity_combo.addItems(["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"])
        self.opacity_combo.setCurrentIndex(3)  # Default 40%
        self.opacity_combo.currentTextChanged.connect(self.change_opacity)

        color_label = QLabel("Color:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["Yellow", "Magenta", "Cyan", "Green"])
        self.color_combo.currentIndexChanged.connect(self.change_color)

        appearance_layout.addWidget(opacity_label)
        appearance_layout.addWidget(self.opacity_combo)
        appearance_layout.addWidget(color_label)
        appearance_layout.addWidget(self.color_combo)
        appearance_group.setLayout(appearance_layout)

        # Add controls to layout
        controls_layout.addWidget(file_group, 0, 0)
        controls_layout.addWidget(point_group, 0, 1)
        controls_layout.addWidget(view_group, 1, 0)
        controls_layout.addWidget(appearance_group, 1, 1)
        controls_layout.addWidget(dicom_group, 2, 0, 1, 2)

        main_layout.addLayout(controls_layout)

        # Add keyboard shortcuts
        self.shortcut_clear = QShortcut(QKeySequence("C"), self)
        self.shortcut_clear.activated.connect(self.clear_points)

        self.shortcut_clear_mask = QShortcut(QKeySequence("M"), self)
        self.shortcut_clear_mask.activated.connect(self.clear_mask)

        self.shortcut_run = QShortcut(QKeySequence("R"), self)
        self.shortcut_run.activated.connect(self.run_inference)

        # Shortcut per zoom e pan
        self.shortcut_reset_zoom = QShortcut(QKeySequence("Z"), self)
        self.shortcut_reset_zoom.activated.connect(self.view.reset_zoom)

        self.shortcut_toggle_pan = QShortcut(QKeySequence("P"), self)
        self.shortcut_toggle_pan.activated.connect(lambda: self.toggle_pan_button.click())

        # Shortcut per navigare tra i frame DICOM
        self.shortcut_prev_frame = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev_frame.activated.connect(self.previous_frame)

        self.shortcut_next_frame = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next_frame.activated.connect(self.next_frame)

        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_action)

        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo_action)

        # Window settings
        self.setLayout(main_layout)
        self.setWindowTitle("CoroSAM Segmentation Tool")
        self.resize(900, 800)

        # Status message
        self.status_label = QLabel("Ready. Load an image or DICOM to begin.")
        main_layout.addWidget(self.status_label)

        # Eraser cursor
        self.update_eraser_cursor()

    def save_state(self, action_description=""):
        """Salva lo stato corrente nella cronologia"""
        state = {
            'tips': self.tips.copy(),
            'branch_points': self.branch_points.copy(),
            'binary_mask': self.binary_mask.copy() if self.binary_mask is not None else None,
            'action': action_description,
            'timestamp': time.time()
        }

        # Se siamo nel mezzo della cronologia, rimuovi gli stati successivi
        if self.current_state_index < len(self.history) - 1:
            self.history = self.history[:self.current_state_index + 1]

        # Aggiungi il nuovo stato
        self.history.append(state)
        self.current_state_index = len(self.history) - 1

        # Mantieni solo gli ultimi max_history stati
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_state_index = len(self.history) - 1

        # Aggiorna il bottone undo
        self.update_undo_button()

    def undo_action(self):
        """Torna indietro di un'azione"""
        if self.current_state_index > 0:
            self.current_state_index -= 1
            self.restore_state(self.history[self.current_state_index])

            action = self.history[self.current_state_index].get('action', 'Unknown')
            self.status_label.setText(f"Undo: {action}")
            self.update_undo_button()
        else:
            self.status_label.setText("No more actions to undo")

    def redo_action(self):
        """Rifai un'azione annullata"""
        if self.current_state_index < len(self.history) - 1:
            self.current_state_index += 1
            self.restore_state(self.history[self.current_state_index])

            action = self.history[self.current_state_index].get('action', 'Unknown')
            self.status_label.setText(f"Redo: {action}")
            self.update_undo_button()
        else:
            self.status_label.setText("No more actions to redo")

    def restore_state(self, state):
        """Ripristina uno stato dalla cronologia"""
        # Ripristina i punti
        self.tips = state['tips'].copy()
        self.branch_points = state['branch_points'].copy()

        # Ripristina la maschera
        if state['binary_mask'] is not None:
            self.binary_mask = state['binary_mask'].copy()
        else:
            self.binary_mask = None

        # Rimuovi i punti visivi esistenti
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items = []

        # Ridisegna i punti
        self.redraw_points()

        # Aggiorna i canali dei punti
        if hasattr(self, 'input_h'):
            self.tips_channel, self.branch_channel = create_point_channels(
                self.tips, self.branch_points, self.input_h)

            if hasattr(self, 'input_image'):
                self.input_image_stack = np.stack(
                    [self.input_image, self.tips_channel, self.branch_channel], axis=0)
                self.input_image_stack_tensor = torch.tensor(
                    self.input_image_stack).float().unsqueeze(0).to(DEVICE)

        # Aggiorna il display
        self.update_display()

    def redraw_points(self):
        """Ridisegna tutti i punti sulla scena"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return

        # Disegna i tips (endpoint)
        for tip in self.tips:
            orig_x = tip[0] / self.input_w * self.original_image.shape[1]
            orig_y = tip[1] / self.input_h * self.original_image.shape[0]

            point_item = self.scene.addEllipse(
                orig_x - POINT_RADIUS,
                orig_y - POINT_RADIUS,
                POINT_RADIUS * 2,
                POINT_RADIUS * 2,
                pen=QPen(POINT_COLORS["Endpoint"], 2),
                brush=QBrush(POINT_COLORS["Endpoint"], Qt.SolidPattern)
            )
            point_item.setZValue(2)
            self.point_items.append(point_item)

        # Disegna i branch points
        for branch in self.branch_points:
            orig_x = branch[0] / self.input_w * self.original_image.shape[1]
            orig_y = branch[1] / self.input_h * self.original_image.shape[0]

            point_item = self.scene.addEllipse(
                orig_x - POINT_RADIUS,
                orig_y - POINT_RADIUS,
                POINT_RADIUS * 2,
                POINT_RADIUS * 2,
                pen=QPen(POINT_COLORS["Intermediate"], 2),
                brush=QBrush(POINT_COLORS["Intermediate"], Qt.SolidPattern)
            )
            point_item.setZValue(2)
            self.point_items.append(point_item)

    def update_undo_button(self):
        """Aggiorna lo stato dei bottoni undo/redo"""
        if hasattr(self, 'undo_button'):
            self.undo_button.setEnabled(self.current_state_index > 0)
        if hasattr(self, 'redo_button'):
            self.redo_button.setEnabled(self.current_state_index < len(self.history) - 1)

    def toggle_pan_mode(self):
        """Toggle between pan mode and point placement mode"""
        is_panning = self.view.toggle_drag_mode()

        if is_panning:
            self.toggle_pan_button.setText("Draw Mode")
            self.status_label.setText("Pan mode: drag to move the view. Click 'Draw Mode' to return to drawing.")
            # Memorize current cursor
            self.prev_cursor = self.view.cursor()
            self.view.setCursor(Qt.OpenHandCursor)
        else:
            self.toggle_pan_button.setText("Pan Mode")
            self.status_label.setText("Draw mode: click to place points.")
            # GO back to previous cursor
            if hasattr(self, 'prev_cursor'):
                self.view.setCursor(self.prev_cursor)
            else:
                self.view.setCursor(Qt.ArrowCursor)
            self.change_point_type(self.curr_point_type)

    def toggle_eraser_mode(self):
        """Toggle eraser mode on/off"""
        is_eraser = self.eraser_mode_check.isChecked()

        if is_eraser:
            self.curr_point_type = "Eraser"
            self.status_label.setText("Eraser mode enabled. Left click + drag to erase.")
            self.update_eraser_cursor()
            # Mostra controlli eraser
            self.eraser_size_label.setVisible(True)
            self.eraser_size_slider.setVisible(True)
        else:
            self.curr_point_type = "Normal"  # Stato normale
            self.status_label.setText("Normal mode. Left click: Endpoint, Right click: Branch point.")
            self.view.setCursor(Qt.ArrowCursor)
            # Nascondi controlli eraser
            self.eraser_size_label.setVisible(False)
            self.eraser_size_slider.setVisible(False)

    def get_eraser_cursor(self, size):
        """Get eraser cursor or create new one"""
        if size not in self.eraser_cursors:
            self.eraser_cursors[size] = create_circle_cursor(size)
        return self.eraser_cursors[size]

    def update_eraser_cursor(self):
        """Update the eraser cursor to match its actual size"""
        if self.curr_point_type == "Eraser":
            # Use the actual eraser size for the cursor
            cursor_size = self.eraser_size * 2  # Multiply by 2 to match the diameter
            self.view.setCursor(self.get_eraser_cursor(cursor_size))

    def change_point_type(self, point_type):
        """Change the current point type (tip, branch or eraser)"""
        self.curr_point_type = point_type
        self.status_label.setText(f"Point type changed to: {point_type}")

        # Change cursor based on point type
        if point_type == "Eraser":
            self.update_eraser_cursor()
        else:
            self.view.setCursor(Qt.ArrowCursor)

    def change_eraser_size(self, size):
        """Change the size of the eraser tool"""
        self.eraser_size = size
        self.status_label.setText(f"Eraser size set to: {size} pixels")

        if self.curr_point_type == "Eraser":
            self.update_eraser_cursor()

    def change_opacity(self, opacity_text):
        """Change the opacity of the mask overlay"""
        self.mask_opacity = float(opacity_text.strip('%')) / 100
        self.update_display()

    def change_color(self, color_index):
        """Change the color of the mask overlay"""
        self.mask_color_idx = color_index
        self.update_display()

    def clear_mask(self):
        """Clear the current segmentation mask"""
        if not hasattr(self, 'original_image'):
            self.status_label.setText("No image loaded.")
            return

        if self.binary_mask is not None and np.any(self.binary_mask):  # Solo se c'è una maschera
            self.save_state("Clear mask")

        # Clear the binary mask
        if self.binary_mask is not None:
            self.binary_mask = np.zeros_like(self.binary_mask)
            self.update_display()
            self.status_label.setText("Mask cleared.")

    def previous_frame(self):
        """Go to previous DICOM frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_spinbox.setValue(self.current_frame)
            self.load_dicom_frame()

    def next_frame(self):
        """Go to next DICOM frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_spinbox.setValue(self.current_frame)
            self.load_dicom_frame()

    def set_frame(self, frame_index):
        """Set DICOM frame manually."""
        if 0 <= frame_index < self.total_frames:
            self.current_frame = frame_index
            self.frame_info_label.setText(f"Frame: {self.current_frame} / {self.total_frames - 1}")
            self.load_dicom_frame()

    def load_image(self):
        """Load an image from file for segmentation."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp *.tif)"
        )

        if not file_path:
            self.status_label.setText("No image selected.")
            return

        try:
            # Read image
            img_np = io.imread(file_path)

            # Convert to RGB if needed
            if len(img_np.shape) == 2:  # Grayscale
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            elif img_np.shape[2] > 3:  # RGBA or similar
                img_3c = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            else:  # Already RGB
                img_3c = img_np

            # Resize for model input
            img_resized = cv2.resize(img_3c, (self.input_w, self.input_h),
                                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)

            # Normalize
            img_norm = (img_resized - img_resized.min()) / np.clip(
                img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
            )

            # Store image data
            self.original_image = img_3c
            self.input_image = img_norm[:, :, 0]  # Use first channel only
            self.image_path = file_path

            # Initialize blank mask
            self.binary_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.current_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

            # Update scene dimensions
            original_h, original_w, _ = self.original_image.shape
            self.scene.setSceneRect(0, 0, original_w, original_h)

            # Set background image
            if self.bg_img is not None:
                self.scene.removeItem(self.bg_img)
            self.bg_img = self.scene.addPixmap(np2pixmap(self.original_image))

            # Clear existing points
            self.clear_points()

            self.status_label.setText(f"Loaded image: {os.path.basename(file_path)}")

            self.history = []
            self.current_state_index = -1
            self.save_state("Load image")

        except Exception as e:
            self.status_label.setText(f"Error loading image: {str(e)}")

    def load_dicom(self):
        """Load DICOM file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose DICOM File", ".", "DICOM Files (*.dcm)"
        )
        self.image_path = file_path

        if not file_path:
            self.status_label.setText("No DICOM file selected.")
            return

        try:
            # Load dicom
            self.dicom_dataset = pydicom.dcmread(file_path)

            if hasattr(self.dicom_dataset, 'NumberOfFrames'):
                self.total_frames = self.dicom_dataset.NumberOfFrames
                self.frame_spinbox.setMaximum(self.total_frames - 1)
                self.frame_info_label.setText(f"Frame: {self.current_frame} / {self.total_frames - 1}")
            else:
                # Single frame
                self.total_frames = 1
                self.frame_spinbox.setMaximum(0)
                self.frame_info_label.setText("Single frame image")

            # Set current frame to 0 and load it
            self.current_frame = 0
            self.frame_spinbox.setValue(0)
            self.play_button.setEnabled(self.total_frames > 1)
            self.load_dicom_frame()

            self.status_label.setText(f"Loaded DICOM file: {os.path.basename(file_path)}")

        except Exception as e:
            self.status_label.setText(f"Error loading DICOM: {str(e)}")

    def load_dicom_frame(self):
        """Load a DICOM frame."""
        if self.dicom_dataset is None:
            return

        try:
            img_arr = self.dicom_dataset.pixel_array[self.current_frame]
            if len(img_arr.shape) == 2:
                img_3c = np.stack([img_arr, img_arr, img_arr], axis=-1)
            else:
                img_3c = img_arr
            # Resize for model input
            img_resized = cv2.resize(img_3c, (self.input_w, self.input_h),
                                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)

            # Normalize
            img_norm = (img_resized - img_resized.min()) / np.clip(
                img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
            )

            # Store image data
            self.original_image = img_3c
            self.input_image = img_norm[:, :, 0]  # Use first channel only

            # Initialize blank mask
            self.binary_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.current_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

            # Update scene dimensions
            original_h, original_w, _ = self.original_image.shape
            self.scene.setSceneRect(0, 0, original_w, original_h)

            # Set background image
            if self.bg_img is not None:
                self.scene.removeItem(self.bg_img)
            self.bg_img = self.scene.addPixmap(np2pixmap(self.original_image))

            # Clear existing points
            self.clear_points()

            self.history = []
            self.current_state_index = -1
            self.save_state("Load image")

        except Exception as e:
            self.status_label.setText(f"Error loading image: {str(e)}")

    def handle_mouse_press(self, event):
        """Handle mouse press events for point placement or eraser"""
        if not hasattr(self, 'original_image'):
            self.status_label.setText("Please load an image first.")
            return

        if self.view.dragMode() == QGraphicsView.ScrollHandDrag:
            return

        # Get click position in scene coordinates
        scene_x, scene_y = event.scenePos().x(), event.scenePos().y()

        if self.eraser_mode_check.isChecked():
            # Modalità eraser: solo tasto sinistro
            if event.button() == Qt.LeftButton and self.binary_mask is not None:
                self.apply_eraser(scene_x, scene_y)
        else:
            # Modalità normale: tasto sinistro = Endpoint, tasto destro = Branch
            if event.button() == Qt.LeftButton:
                self.add_specific_point(scene_x, scene_y, "Endpoint")
            elif event.button() == Qt.RightButton:
                self.add_specific_point(scene_x, scene_y, "Intermediate")

    def add_specific_point(self, scene_x, scene_y, point_type):
        """Add a specific type of point at the given scene coordinates"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return

        # Scale to normalized coordinates for the model
        scaled_x = scene_x / self.original_image.shape[1] * self.input_w
        scaled_y = scene_y / self.original_image.shape[0] * self.input_h

        self.save_state(f"Add {point_type} point")

        # Add point based on specified type
        if point_type == "Endpoint":
            self.tips.append((scaled_x, scaled_y))
        elif point_type == "Intermediate":  # branch
            self.branch_points.append((scaled_x, scaled_y))

        # Update point channels for model input
        self.tips_channel, self.branch_channel = create_point_channels(
            self.tips, self.branch_points, self.input_h)

        # Stack input channels for model
        self.input_image_stack = np.stack(
            [self.input_image, self.tips_channel, self.branch_channel], axis=0)
        self.input_image_stack_tensor = torch.tensor(self.input_image_stack).float().unsqueeze(0).to(DEVICE)

        # Add visual point to scene
        point_color = POINT_COLORS[point_type]
        point_item = self.scene.addEllipse(
            scene_x - POINT_RADIUS,
            scene_y - POINT_RADIUS,
            POINT_RADIUS * 2,
            POINT_RADIUS * 2,
            pen=QPen(point_color, 2),
            brush=QBrush(point_color, Qt.SolidPattern)
        )
        point_item.setZValue(2)  # Make points appear above mask
        self.point_items.append(point_item)

        # Update display
        self.status_label.setText(f"Added {point_type} point at ({int(scene_x)}, {int(scene_y)})")
        self.update_display()

        # Run inference automatically if enabled
        if self.auto_inference_check.isChecked():
            self.run_inference()

    def handle_mouse_move(self, last_pos, curr_pos):
        """Handle mouse move events for continuous erasing"""
        if not hasattr(self, 'original_image') or self.binary_mask is None:
            return

        if not hasattr(self, '_eraser_session_saved'):
            self.save_state("Eraser operation")
            self._eraser_session_saved = True

        # Erasing solo in modalità eraser
        if self.eraser_mode_check.isChecked():
            pts = self.interpolate_points(last_pos.x(), last_pos.y(), curr_pos.x(), curr_pos.y())
            for x, y in pts:
                self.apply_eraser(x, y)

    def handle_mouse_release(self, event):
        """Handle mouse release events"""
        # Reset della sessione eraser
        if hasattr(self, '_eraser_session_saved'):
            delattr(self, '_eraser_session_saved')

    def interpolate_points(self, x1, y1, x2, y2):
        """Interpolate points between two coordinates for smooth line drawing"""
        points = []
        # Calculate distance
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # If distance is small, return just the end point
        if dist < 1:
            return [(x2, y2)]

        # Calculate number of points to interpolate
        steps = int(dist / (self.eraser_size / 4))
        steps = max(1, steps)  # At least 1 step

        # Interpolate
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append((x, y))

        return points

    def apply_eraser(self, scene_x, scene_y):
        """Apply eraser at the given scene coordinates"""
        if self.binary_mask is None:
            return

        # Convert scene coordinates to original image coordinates
        img_h, img_w = self.original_image.shape[:2]

        # Ensure scene coordinates are within bounds
        scene_x = max(0, min(scene_x, img_w - 1))
        scene_y = max(0, min(scene_y, img_h - 1))

        # Calculate eraser area
        eraser_radius = self.eraser_size
        y_min = max(0, int(scene_y - eraser_radius))
        y_max = min(img_h, int(scene_y + eraser_radius))
        x_min = max(0, int(scene_x - eraser_radius))
        x_max = min(img_w, int(scene_x + eraser_radius))

        # Create a circular mask for eraser
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - scene_x) ** 2 + (y - scene_y) ** 2 <= eraser_radius ** 2

        # Apply eraser (set mask values to 0)
        self.binary_mask[y_min:y_max, x_min:x_max][mask] = 0

        # Update display
        self.update_display()

    def add_point(self, scene_x, scene_y):
        """Add a point (tip or branch) at the given scene coordinates"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return

        # Scale to normalized coordinates for the model
        scaled_x = scene_x / self.original_image.shape[1] * self.input_w
        scaled_y = scene_y / self.original_image.shape[0] * self.input_h

        # Add point based on current type selection
        if self.curr_point_type == "Endpoint":
            self.tips.append((scaled_x, scaled_y))
        elif self.curr_point_type == "Intermediate":  # branch
            self.branch_points.append((scaled_x, scaled_y))

        # Update point channels for model input
        self.tips_channel, self.branch_channel = create_point_channels(
            self.tips, self.branch_points, self.input_h)

        # Stack input channels for model
        self.input_image_stack = np.stack(
            [self.input_image, self.tips_channel, self.branch_channel], axis=0)
        self.input_image_stack_tensor = torch.tensor(self.input_image_stack).float().unsqueeze(0).to(DEVICE)

        # Add visual point to scene
        point_color = POINT_COLORS[self.curr_point_type]
        point_item = self.scene.addEllipse(
            scene_x - POINT_RADIUS,
            scene_y - POINT_RADIUS,
            POINT_RADIUS * 2,
            POINT_RADIUS * 2,
            pen=QPen(point_color, 2),
            brush=QBrush(point_color, Qt.SolidPattern)
        )
        point_item.setZValue(2)  # Make points appear above mask
        self.point_items.append(point_item)

        # Update display
        self.status_label.setText(f"Added {self.curr_point_type} point at ({int(scene_x)}, {int(scene_y)})")
        self.update_display()

        # Run inference automatically if enabled
        if self.auto_inference_check.isChecked():
            self.run_inference()

    def update_display(self):
        """Update the display with current mask and points"""
        if not hasattr(self, 'original_image'):
            return

        # Start with the original image
        display_img = self.original_image.copy()

        # Add mask overlay if exists
        if self.binary_mask is not None and np.max(self.binary_mask) > 0:
            mask_overlay = np.zeros_like(display_img)
            mask_color = MASK_COLORS[self.mask_color_idx % len(MASK_COLORS)]
            mask_overlay[self.binary_mask > 0] = mask_color
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, self.mask_opacity, 0)

        point_overlay = np.zeros_like(display_img)

        # Draw tips (green) on the overlay
        for tip in self.tips:
            # Convert normalized coordinates back to original image space
            orig_x = int(tip[0] / self.input_w * self.original_image.shape[1])
            orig_y = int(tip[1] / self.input_h * self.original_image.shape[0])

            # Draw a circle for each tip point
            cv2.circle(point_overlay, (orig_x, orig_y), POINT_RADIUS, (0, 255, 0), -1)

        # Draw branch points (blue) on the overlay
        for branch in self.branch_points:
            # Convert normalized coordinates back to original image space
            orig_x = int(branch[0] / self.input_w * self.original_image.shape[1])
            orig_y = int(branch[1] / self.input_h * self.original_image.shape[0])

            # Draw a circle for each branch point
            cv2.circle(point_overlay, (orig_x, orig_y), POINT_RADIUS, (0, 0, 255), -1)

        # Blend the point overlay
        display_img = cv2.addWeighted(display_img, 1.0, point_overlay, 0.7, 0)

        # Update display
        if self.bg_img is not None:
            self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(display_img))

    def clear_points(self):
        """Clear all annotation points"""
        if self.tips or self.branch_points:
            self.save_state("Clear points")

        # Clear data structures
        self.tips = []
        self.branch_points = []

        # Remove visual point items from scene
        if hasattr(self, 'scene'):
            for item in self.point_items:
                self.scene.removeItem(item)
        self.point_items = []

        # Reset point channels
        if hasattr(self, 'input_h'):
            self.tips_channel = np.zeros((self.input_h, self.input_h), dtype=np.float32)
            self.branch_channel = np.zeros((self.input_h, self.input_h), dtype=np.float32)

            if hasattr(self, 'input_image'):
                # Recreate input stack
                self.input_image_stack = np.stack(
                    [self.input_image, self.tips_channel, self.branch_channel], axis=0)
                self.input_image_stack_tensor = torch.tensor(
                    self.input_image_stack).float().unsqueeze(0).to(DEVICE)

        # Update display
        self.update_display()
        self.status_label.setText("All points cleared.")

    def run_inference(self):
        """Run model inference to update segmentation mask"""
        if not hasattr(self, 'input_image_stack_tensor'):
            self.status_label.setText("Please load an image and add points first.")
            return

        self.save_state("Run inference")

        try:
            # Run inference with no gradients for efficiency
            with torch.no_grad():
                self.status_label.setText("Running inference...")
                self.model_prediction = torch.sigmoid(model(self.input_image_stack_tensor))

            # Extract probabilities and create binary mask
            probs = self.model_prediction.squeeze().detach().cpu().numpy()

            # Resize to original image dimensions
            resized_probs = cv2.resize(
                probs,
                (self.original_image.shape[1], self.original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Apply threshold to get binary mask
            resized_new_mask = (resized_probs > 0.5).astype(np.uint8)

            # Merge the new mask with the existing mask instead of replacing it
            if self.binary_mask is None:
                self.binary_mask = resized_new_mask
            else:
                # Logical OR to combine masks
                self.binary_mask = np.logical_or(self.binary_mask, resized_new_mask).astype(np.uint8)

            # Update the display with the new mask
            self.update_display()
            self.status_label.setText("Inference complete. Mask updated.")

        except Exception as e:
            self.status_label.setText(f"Error during inference: {str(e)}")

    def save_mask(self):
        """Save the current binary mask to a file"""
        if not hasattr(self, 'binary_mask'):
            self.status_label.setText("No mask to save.")
            return

        try:
            # Ensure we save only the binary segmentation mask without points
            binary_mask_to_save = (self.binary_mask > 0).astype(np.uint8) * 255

            # Allow user to choose save location and filename
            default_path = f"{os.path.splitext(self.image_path)[0]}_mask.png"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Segmentation Mask", default_path, "PNG Files (*.png)"
            )

            if save_path:
                io.imsave(save_path, binary_mask_to_save)
                self.status_label.setText(f"Saved mask to {os.path.basename(save_path)}")
            else:
                self.status_label.setText("Mask saving cancelled.")

        except Exception as e:
            self.status_label.setText(f"Error saving mask: {str(e)}")

    def toggle_playback(self):
        """Toggle play/pause of automatic frame advancement"""
        if self.play_button.isChecked():
            self.start_playback()
        else:
            self.stop_playback()

    def start_playback(self):
        """Start automatic playback of DICOM frames"""
        if self.total_frames <= 1:
            self.play_button.setChecked(False)
            return

        # If we're at the last frame, go back to the beginning
        if self.current_frame >= self.total_frames - 1:
            self.current_frame = 0
            self.frame_spinbox.setValue(0)
            self.load_dicom_frame()

        interval = int(1000 / self.playback_fps)  # Convert fps to milliseconds
        self.playback_timer.start(interval)
        self.play_button.setText("Pause")
        self.status_label.setText("Playing DICOM series")

        # Disable frame controls during playback
        self.frame_spinbox.setEnabled(False)

    def stop_playback(self):
        """Stop automatic playback of DICOM frames"""
        self.playback_timer.stop()
        self.play_button.setText("Play")
        self.status_label.setText("Playback paused")

        # Re-enable frame controls
        self.frame_spinbox.setEnabled(True)

    def advance_frame(self):
        """Advance to the next frame during playback"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_spinbox.setValue(self.current_frame)
            self.load_dicom_frame()
        else:
            # We've reached the end, restart from the beginning
            self.current_frame = 0
            self.frame_spinbox.setValue(self.current_frame)
            self.load_dicom_frame()


# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationWindow()
    window.show()
    sys.exit(app.exec_())
