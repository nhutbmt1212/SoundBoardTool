"""
Modern Soundboard GUI with beautiful design
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os

# Ensure imports work from src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soundboard import Soundboard
from config import Config

class ModernButton(tk.Canvas):
    """Custom modern button with gradient and hover effects"""
    def __init__(self, parent, text, command, **kwargs):
        super().__init__(parent, **kwargs)
        self.text = text
        self.command = command
        self.is_hovered = False
        self.is_pressed = False
        
        # Colors
        self.bg_normal = "#6366f1"
        self.bg_hover = "#4f46e5"
        self.bg_pressed = "#4338ca"
        self.text_color = "#ffffff"
        
        self.bind("<Button-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
        self.draw()
    
    def draw(self):
        self.delete("all")
        width = self.winfo_width() or 150
        height = self.winfo_height() or 60
        
        # Determine color based on state
        if self.is_pressed:
            color = self.bg_pressed
        elif self.is_hovered:
            color = self.bg_hover
        else:
            color = self.bg_normal
        
        # Draw rounded rectangle
        radius = 12
        self.create_rounded_rect(2, 2, width-2, height-2, radius, fill=color, outline="")
        
        # Draw text
        self.create_text(width//2, height//2, text=self.text, 
                        fill=self.text_color, font=("Segoe UI", 11, "bold"))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_press(self, event):
        self.is_pressed = True
        self.draw()
    
    def on_release(self, event):
        self.is_pressed = False
        self.draw()
        if self.command:
            self.command()
    
    def on_enter(self, event):
        self.is_hovered = True
        self.draw()
    
    def on_leave(self, event):
        self.is_hovered = False
        self.draw()

class SoundboardUI:
    def __init__(self):
        self.config = Config()
        self.soundboard = Soundboard(self.config.sounds_dir)
        self.root = tk.Tk()
        self.root.title("🎵 Soundboard Pro")
        self.root.geometry("900x700")
        self.root.configure(bg="#0f172a")
        
        # Theme colors
        self.bg_primary = "#0f172a"
        self.bg_secondary = "#1e293b"
        self.bg_tertiary = "#334155"
        self.accent = "#6366f1"
        self.text_primary = "#f1f5f9"
        self.text_secondary = "#94a3b8"
        
        self.volume = tk.DoubleVar(value=self.config.default_volume * 100)
        
        # Routing status label
        self.routing_status_label = None
        
        self.setup_ui()
        
        # Cập nhật status dựa trên VB-Cable detection
        self._update_vb_cable_status()
    
    def setup_ui(self):
        """Setup modern UI"""
        # Header
        header = tk.Frame(self.root, bg=self.bg_secondary, height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header, 
            text="🎵 Soundboard Pro",
            font=("Segoe UI", 24, "bold"),
            bg=self.bg_secondary,
            fg=self.text_primary
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)
        
        # Routing status - hiển thị trạng thái kết nối
        self.routing_status_label = tk.Label(
            header,
            text="🔄 Đang kết nối...",
            font=("Segoe UI", 10),
            bg=self.bg_secondary,
            fg=self.text_secondary
        )
        self.routing_status_label.pack(side=tk.RIGHT, padx=30, pady=20)
        
        # Control panel
        control_frame = tk.Frame(self.root, bg=self.bg_primary)
        control_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Left controls
        left_controls = tk.Frame(control_frame, bg=self.bg_primary)
        left_controls.pack(side=tk.LEFT)
        
        self.create_control_button(left_controls, "➕ Add Sound", self.add_sound).pack(side=tk.LEFT, padx=5)
        self.create_control_button(left_controls, "⏹️ Stop All", self.stop_all).pack(side=tk.LEFT, padx=5)
        self.create_control_button(left_controls, "🔄 Refresh", self.refresh_sounds).pack(side=tk.LEFT, padx=5)
        
        # Right controls - Volume
        right_controls = tk.Frame(control_frame, bg=self.bg_primary)
        right_controls.pack(side=tk.RIGHT)
        
        vol_label = tk.Label(
            right_controls,
            text="🔊 Volume:",
            font=("Segoe UI", 10),
            bg=self.bg_primary,
            fg=self.text_primary
        )
        vol_label.pack(side=tk.LEFT, padx=5)
        
        volume_slider = tk.Scale(
            right_controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume,
            command=self.on_volume_change,
            bg=self.bg_secondary,
            fg=self.text_primary,
            highlightthickness=0,
            troughcolor=self.bg_tertiary,
            activebackground=self.accent,
            length=150
        )
        volume_slider.set(70)
        volume_slider.pack(side=tk.LEFT, padx=5)
        
        # Main content area with canvas for scrolling
        content_container = tk.Frame(self.root, bg=self.bg_primary)
        content_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(content_container, bg=self.bg_primary, highlightthickness=0)
        scrollbar = tk.Scrollbar(content_container, orient="vertical", command=canvas.yview)
        
        self.main_frame = tk.Frame(canvas, bg=self.bg_primary)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_frame = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_frame, width=event.width)
        
        self.main_frame.bind("<Configure>", configure_scroll)
        canvas.bind("<Configure>", configure_scroll)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Footer
        footer = tk.Frame(self.root, bg=self.bg_secondary, height=40)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        footer_text = tk.Label(
            footer,
            text="Made with ❤️ | Press buttons to play sounds",
            font=("Segoe UI", 9),
            bg=self.bg_secondary,
            fg=self.text_secondary
        )
        footer_text.pack(pady=10)
        
        self.refresh_sounds()
    
    def create_control_button(self, parent, text, command):
        """Create a control button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10),
            bg=self.bg_tertiary,
            fg=self.text_primary,
            activebackground=self.accent,
            activeforeground=self.text_primary,
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            borderwidth=0
        )
        return btn
    
    def refresh_sounds(self):
        """Refresh sound buttons with modern design"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.soundboard.load_sounds()
        sounds = self.soundboard.get_sound_list()
        
        if not sounds:
            empty_label = tk.Label(
                self.main_frame,
                text="📂 No sounds found\n\nClick 'Add Sound' to get started!",
                font=("Segoe UI", 14),
                bg=self.bg_primary,
                fg=self.text_secondary,
                justify=tk.CENTER
            )
            empty_label.pack(expand=True, pady=100)
            return
        
        # Create grid of sound buttons
        row, col = 0, 0
        max_cols = 4
        
        for sound_name in sounds:
            btn_frame = tk.Frame(self.main_frame, bg=self.bg_primary)
            btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Sound button
            btn = ModernButton(
                btn_frame,
                text=sound_name,
                command=lambda name=sound_name: self.play_sound(name),
                width=180,
                height=60,
                bg=self.bg_primary,
                highlightthickness=0
            )
            btn.pack(fill=tk.BOTH, expand=True)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(max_cols):
            self.main_frame.grid_columnconfigure(i, weight=1)
    
    def play_sound(self, sound_name):
        """Play sound with visual feedback"""
        self.soundboard.play_sound(sound_name)
    
    def stop_all(self):
        """Stop all sounds"""
        self.soundboard.stop_all()
    
    def on_volume_change(self, value):
        """Handle volume change"""
        volume = float(value) / 100
        self.soundboard.set_volume(volume)
    
    def add_sound(self):
        """Add new sound with modern dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.flac"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            if self.soundboard.add_sound(file_path):
                self.refresh_sounds()
                messagebox.showinfo("✅ Success", "Sound added successfully!")
            else:
                messagebox.showerror("❌ Error", "Failed to add sound")
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def _update_vb_cable_status(self):
        """Cập nhật status dựa trên VB-Cable detection từ soundboard"""
        if self.soundboard.is_vb_cable_connected():
            self._update_routing_status(
                "🎙️ Discord: Chọn 'CABLE Output' làm Input",
                "#10b981"
            )
        else:
            self._update_routing_status(
                "⚠️ VB-Cable chưa cài - Tải tại vb-audio.com/Cable",
                "#f59e0b"
            )
    
    def _update_routing_status(self, text, color):
        """Cập nhật status label trên UI chính"""
        if self.routing_status_label:
            self.routing_status_label.config(text=text, fg=color)
    
    def on_closing(self):
        """Cleanup on window close"""
        if hasattr(self.soundboard, 'cleanup'):
            self.soundboard.cleanup()
        self.root.destroy()
