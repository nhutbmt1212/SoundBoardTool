"""
Modern Soundboard GUI with PySide6
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QSlider, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon

from soundboard import Soundboard
from config import Config


class SoundButton(QPushButton):
    """Custom styled sound button"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(160, 60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QPushButton:pressed {
                background-color: #4338ca;
            }
        """)


class ControlButton(QPushButton):
    """Control button style"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f1f5f9;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                padding: 10px 18px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:pressed {
                background-color: #6366f1;
            }
        """)


class SoundboardUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.soundboard = Soundboard(self.config.sounds_dir)
        
        # Set initial volume từ config
        self.soundboard.set_volume(self.config.default_volume)
        
        self.setWindowTitle("🎵 Soundboard Pro")
        self.setMinimumSize(900, 700)
        self.setup_ui()
        self._update_vb_cable_status()
    
    def setup_ui(self):
        """Setup modern UI"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #f1f5f9;
            }
        """)
        
        # Header
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: #1e293b;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 30, 0)
        
        title = QLabel("🎵 Soundboard Pro")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #f1f5f9; background: transparent;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("🔄 Đang kết nối...")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #94a3b8; background: transparent;")
        header_layout.addWidget(self.status_label)
        
        main_layout.addWidget(header)
        
        # Control panel
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #0f172a;")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 15, 20, 15)
        
        # Left controls
        btn_add = ControlButton("➕ Add Sound")
        btn_add.clicked.connect(self.add_sound)
        control_layout.addWidget(btn_add)
        
        btn_stop = ControlButton("⏹️ Stop All")
        btn_stop.clicked.connect(self.stop_all)
        control_layout.addWidget(btn_stop)
        
        btn_refresh = ControlButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh_sounds)
        control_layout.addWidget(btn_refresh)
        
        control_layout.addStretch()
        
        # Volume control
        vol_label = QLabel("🔊 Volume:")
        vol_label.setStyleSheet("color: #f1f5f9; background: transparent;")
        control_layout.addWidget(vol_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.valueChanged.connect(self.on_volume_change)
        # Set value sau khi connect để trigger on_volume_change
        self.volume_slider.setValue(int(self.config.default_volume * 100))
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #334155;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 4px;
            }
        """)
        control_layout.addWidget(self.volume_slider)
        
        self.vol_value = QLabel(f"{int(self.config.default_volume * 100)}%")
        self.vol_value.setStyleSheet("color: #94a3b8; background: transparent; min-width: 40px;")
        control_layout.addWidget(self.vol_value)
        
        main_layout.addWidget(control_frame)
        
        # Scroll area for sound buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #0f172a;
            }
            QScrollBar:vertical {
                background: #1e293b;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                border-radius: 5px;
            }
        """)
        
        self.sounds_container = QWidget()
        self.sounds_layout = QGridLayout(self.sounds_container)
        self.sounds_layout.setContentsMargins(20, 10, 20, 10)
        self.sounds_layout.setSpacing(15)
        scroll.setWidget(self.sounds_container)
        
        main_layout.addWidget(scroll, 1)
        
        # Footer
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet("background-color: #1e293b;")
        footer_layout = QHBoxLayout(footer)
        
        footer_text = QLabel("Made with ❤️ | Press buttons to play sounds")
        footer_text.setFont(QFont("Segoe UI", 9))
        footer_text.setStyleSheet("color: #94a3b8; background: transparent;")
        footer_text.setAlignment(Qt.AlignCenter)
        footer_layout.addWidget(footer_text)
        
        main_layout.addWidget(footer)
        
        self.refresh_sounds()

    
    def refresh_sounds(self):
        """Refresh sound buttons"""
        # Clear existing buttons
        while self.sounds_layout.count():
            item = self.sounds_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.soundboard.load_sounds()
        sounds = self.soundboard.get_sound_list()
        
        if not sounds:
            empty_label = QLabel("📂 No sounds found\n\nClick 'Add Sound' to get started!")
            empty_label.setFont(QFont("Segoe UI", 14))
            empty_label.setStyleSheet("color: #94a3b8; background: transparent;")
            empty_label.setAlignment(Qt.AlignCenter)
            self.sounds_layout.addWidget(empty_label, 0, 0, 1, 4)
            return
        
        row, col = 0, 0
        max_cols = 4
        
        for sound_name in sounds:
            btn = SoundButton(sound_name)
            btn.clicked.connect(lambda checked, name=sound_name: self.play_sound(name))
            self.sounds_layout.addWidget(btn, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def play_sound(self, sound_name):
        """Play sound"""
        self.soundboard.play_sound(sound_name)
    
    def stop_all(self):
        """Stop all sounds"""
        self.soundboard.stop_all()
    
    def on_volume_change(self, value):
        """Handle volume change"""
        volume = value / 100
        self.soundboard.set_volume(volume)
        self.vol_value.setText(f"{value}%")
    
    def add_sound(self):
        """Add new sound"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            if self.soundboard.add_sound(file_path):
                self.refresh_sounds()
                QMessageBox.information(self, "Success", "Sound added successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to add sound")
    
    def _update_vb_cable_status(self):
        """Update VB-Cable status"""
        if self.soundboard.is_vb_cable_connected():
            self.status_label.setText("🎙️ Discord: Chọn 'CABLE Output' làm Input")
            self.status_label.setStyleSheet("color: #10b981; background: transparent;")
        else:
            self.status_label.setText("⚠️ VB-Cable chưa cài - Tải tại vb-audio.com/Cable")
            self.status_label.setStyleSheet("color: #f59e0b; background: transparent;")
    
    def closeEvent(self, event):
        """Cleanup on close"""
        if hasattr(self.soundboard, 'cleanup'):
            self.soundboard.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = SoundboardUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
