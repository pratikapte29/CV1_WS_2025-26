import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# you can use GUI-related libraries if needed
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# TODO: Your implementation here

from graphcut_core import GraphCutCore, compute_iou


class ImageCanvas(tk.Canvas):
    def __init__(self, parent, gui_parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.gui_parent = gui_parent
        self.config(bg='white', highlightthickness=2, highlightbackground='black')
        
        self.drawing = False
        self.last_point = None
        
        self.bind("<Button-1>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def on_mouse_down(self, event):
        if self.gui_parent.image is not None:
            self.drawing = True
            self.gui_parent.save_to_history()
            pos = self.map_to_image(event.x, event.y)
            if pos:
                self.last_point = pos
                self.gui_parent.draw_at(pos)
                
    def on_mouse_drag(self, event):
        if self.drawing and self.gui_parent.image is not None:
            pos = self.map_to_image(event.x, event.y)
            if pos:
                if self.last_point:
                    self.gui_parent.draw_line(self.last_point, pos)
                self.last_point = pos
                
    def on_mouse_up(self, event):
        self.drawing = False
        self.last_point = None
        
    def map_to_image(self, canvas_x, canvas_y):
        if self.gui_parent.image is None or self.gui_parent.photo_image is None:
            return None
            
        h, w = self.gui_parent.image.shape[:2]
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        
        # Calculate scale and offset
        scale = min(canvas_w / w, canvas_h / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        offset_x = (canvas_w - scaled_w) // 2
        offset_y = (canvas_h - scaled_h) // 2
        
        # Map canvas coordinates to image coordinates
        img_x = int((canvas_x - offset_x) / scale)
        img_y = int((canvas_y - offset_y) / scale)
        
        if 0 <= img_x < w and 0 <= img_y < h:
            return (img_x, img_y)
        return None


class InteractiveSegmentationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Interactive Graph Cut Segmentation")
        self.root.geometry("1200x800")
        
        # Data
        self.image = None  # Original image (BGR)
        self.display_image = None  # Image with scribbles
        self.scribble_mask = None  # 0=unlabeled, 1=fg, 2=bg
        self.segmentation_mask = None
        self.ground_truth = None
        self.photo_image = None  # For tkinter display
        
        # Drawing state
        self.brush_size = 5
        self.current_label = 1  # 1=fg, 2=bg
        
        # History for undo
        self.history = []
        self.max_history = 20
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = tk.Frame(main_frame, width=300, bg='lightgray')
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # Load Image Button
        load_btn = tk.Button(control_frame, text="Load Image", 
                            command=self.load_image,
                            width=25, height=2, bg='lightblue', 
                            font=('Arial', 10, 'bold'))
        load_btn.pack(pady=5)
        
        # Load Ground Truth Button
        load_gt_btn = tk.Button(control_frame, text="Load Ground Truth (Optional)",
                               command=self.load_ground_truth,
                               width=25, bg='lightgray')
        load_gt_btn.pack(pady=5)
        
        # Drawing Mode
        mode_frame = tk.LabelFrame(control_frame, text="Drawing Mode", 
                                  bg='lightgray', font=('Arial', 10, 'bold'))
        mode_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.mode_var = tk.IntVar(value=1)
        
        fg_radio = tk.Radiobutton(mode_frame, text="Foreground (White)",
                                 variable=self.mode_var, value=1,
                                 command=self.change_mode, bg='lightgray')
        fg_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        bg_radio = tk.Radiobutton(mode_frame, text="Background (Red)",
                                 variable=self.mode_var, value=2,
                                 command=self.change_mode, bg='lightgray')
        bg_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Brush Size
        brush_frame = tk.LabelFrame(control_frame, text="Brush Size",
                                   bg='lightgray', font=('Arial', 10, 'bold'))
        brush_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.brush_label = tk.Label(brush_frame, text=f"Size: {self.brush_size}",
                                   bg='lightgray')
        self.brush_label.pack(pady=2)
        
        self.brush_slider = tk.Scale(brush_frame, from_=1, to=20,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_brush_size,
                                    bg='lightgray', highlightthickness=0)
        self.brush_slider.set(5)
        self.brush_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # Model Selection
        model_frame = tk.LabelFrame(control_frame, text="Color Model",
                                   bg='lightgray', font=('Arial', 10, 'bold'))
        model_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.model_var = tk.StringVar(value="gmm")
        
        gmm_radio = tk.Radiobutton(model_frame, text="GMM",
                                  variable=self.model_var, value="gmm",
                                  bg='lightgray')
        gmm_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        hist_radio = tk.Radiobutton(model_frame, text="Histogram",
                                   variable=self.model_var, value="histogram",
                                   bg='lightgray')
        hist_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # GMM Components
        gmm_frame = tk.LabelFrame(control_frame, text="GMM Components",
                                 bg='lightgray')
        gmm_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.gmm_label = tk.Label(gmm_frame, text="Components: 5", bg='lightgray')
        self.gmm_label.pack(pady=2)
        
        self.gmm_slider = tk.Scale(gmm_frame, from_=2, to=10,
                                  orient=tk.HORIZONTAL,
                                  command=lambda v: self.gmm_label.config(
                                      text=f"Components: {v}"),
                                  bg='lightgray', highlightthickness=0)
        self.gmm_slider.set(5)
        self.gmm_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # Histogram Bins
        bins_frame = tk.LabelFrame(control_frame, text="Histogram Bins",
                                  bg='lightgray')
        bins_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.bins_label = tk.Label(bins_frame, text="Bins: 16", bg='lightgray')
        self.bins_label.pack(pady=2)
        
        self.bins_slider = tk.Scale(bins_frame, from_=8, to=64,
                                   orient=tk.HORIZONTAL,
                                   command=lambda v: self.bins_label.config(
                                       text=f"Bins: {v}"),
                                   bg='lightgray', highlightthickness=0)
        self.bins_slider.set(16)
        self.bins_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # Beta Parameter
        beta_frame = tk.LabelFrame(control_frame, text="Beta (×0.001)",
                                  bg='lightgray')
        beta_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.beta_label = tk.Label(beta_frame, text="β: 0.005", bg='lightgray')
        self.beta_label.pack(pady=2)
        
        self.beta_slider = tk.Scale(beta_frame, from_=1, to=500,
                                   orient=tk.HORIZONTAL, resolution=1,
                                   command=lambda v: self.beta_label.config(
                                       text=f"β: {float(v)/1000:.3f}"),
                                   bg='lightgray', highlightthickness=0)
        self.beta_slider.set(5)
        self.beta_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # Lambda Parameter
        lambda_frame = tk.LabelFrame(control_frame, text="Lambda (Smoothness)",
                                    bg='lightgray')
        lambda_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.lambda_label = tk.Label(lambda_frame, text="λ: 100", bg='lightgray')
        self.lambda_label.pack(pady=2)
        
        self.lambda_slider = tk.Scale(lambda_frame, from_=10, to=300,
                                     orient=tk.HORIZONTAL,
                                     command=lambda v: self.lambda_label.config(
                                         text=f"λ: {v}"),
                                     bg='lightgray', highlightthickness=0)
        self.lambda_slider.set(100)
        self.lambda_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # Action Buttons
        segment_btn = tk.Button(control_frame, text="Run Segmentation\n(Space)",
                               command=self.run_segmentation,
                               width=25, height=2, bg='lightgreen',
                               font=('Arial', 10, 'bold'))
        segment_btn.pack(pady=10)
        
        undo_btn = tk.Button(control_frame, text="Undo (Ctrl+Z)",
                            command=self.undo, width=25,
                            bg='lightyellow')
        undo_btn.pack(pady=5)
        
        reset_btn = tk.Button(control_frame, text="Reset All (R)",
                             command=self.reset, width=25,
                             bg='orange')
        reset_btn.pack(pady=5)
        
        save_btn = tk.Button(control_frame, text="Save Mask (S)",
                            command=self.save_mask, width=25,
                            bg='lightcoral')
        save_btn.pack(pady=5)
        
        # IoU Display
        self.iou_label = tk.Label(control_frame, text="IoU: N/A",
                                 bg='white', relief=tk.SOLID, borderwidth=2,
                                 font=('Arial', 14, 'bold'), height=2)
        self.iou_label.pack(pady=10, padx=10, fill=tk.X)
        
        # Right panel - Canvas
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = ImageCanvas(canvas_frame, self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        self.root.bind("<space>", lambda e: self.run_segmentation())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("r", lambda e: self.reset())
        self.root.bind("R", lambda e: self.reset())
        self.root.bind("s", lambda e: self.save_mask())
        self.root.bind("S", lambda e: self.save_mask())
        
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath:
            return
            
        self.image = cv2.imread(filepath)
        
        h, w = self.image.shape[:2]
        self.scribble_mask = np.zeros((h, w), dtype=np.uint8)
        self.segmentation_mask = None
        self.history = []
        self.display_image = self.image.copy()
        
        self.update_canvas()
        
    def load_ground_truth(self):
        filepath = filedialog.askopenfilename(
            title="Open Ground Truth",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath:
            return
            
        self.ground_truth = cv2.imread(filepath, 0)
        self.ground_truth = (self.ground_truth > 0).astype(np.uint8)
        messagebox.showinfo("Success", "Ground truth loaded!")
        
    def change_mode(self):
        self.current_label = self.mode_var.get()
        
    def update_brush_size(self, value):
        self.brush_size = int(float(value))
        self.brush_label.config(text=f"Size: {self.brush_size}")
        
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
            messagebox.showwarning("Undo", "No more undo history!")
            
    def reset(self):
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.scribble_mask = np.zeros((h, w), dtype=np.uint8)
            self.segmentation_mask = None
            self.history = []
            self.display_image = self.image.copy()
            self.update_canvas()
            self.iou_label.config(text="IoU: N/A")
            
    def draw_at(self, pos):
        x, y = pos
        cv2.circle(self.scribble_mask, (x, y), self.brush_size, 
                  self.current_label, -1)
        self.update_display()
        self.update_canvas()
        
    def draw_line(self, p1, p2):
        cv2.line(self.scribble_mask, p1, p2, self.current_label, 
                self.brush_size * 2)
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
            overlay[self.segmentation_mask == 1] = [0, 255, 0]  # BGR green
            self.display_image = cv2.addWeighted(
                self.display_image, 0.7, overlay, 0.3, 0)
            
    def update_canvas(self):
        if self.display_image is not None:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Get canvas size
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            # Make sure canvas has been rendered
            if canvas_w <= 1:
                canvas_w = 800
            if canvas_h <= 1:
                canvas_h = 600
            
            # Calculate scaled size maintaining aspect ratio
            img_w, img_h = pil_image.size
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # Resize image
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            
            # Center the image
            x_offset = (canvas_w - new_w) // 2
            y_offset = (canvas_h - new_h) // 2
            
            self.canvas.create_image(x_offset, y_offset, 
                                    image=self.photo_image, anchor=tk.NW)
            
    def run_segmentation(self):
        if self.image is None:
            messagebox.showwarning("Error", "Please load an image first!")
            return
            
        if np.sum(self.scribble_mask == 1) == 0 or np.sum(self.scribble_mask == 2) == 0:
            messagebox.showwarning("Error", 
                "Please draw both foreground and background scribbles!")
            return
            
        try:
            # Get parameters
            use_gmm = (self.model_var.get() == "gmm")
            beta = self.beta_slider.get() / 1000.0
            lam = self.lambda_slider.get()
            n_components = self.gmm_slider.get()
            bins = self.bins_slider.get()
            
            # Create GraphCutCore instance
            gc = GraphCutCore(self.image, self.scribble_mask)
            
            # Build color models
            if use_gmm:
                gc.build_color_models_gmm(n_components=n_components)
            else:
                gc.build_color_models_histogram(bins=bins)
            
            # Run graph cut
            self.segmentation_mask = gc.graph_cut(use_gmm=use_gmm, 
                                                 beta=beta, lam=lam)
            
            # Update display
            self.update_display()
            self.update_canvas()
            
            # Compute IoU if GT available
            if self.ground_truth is not None:
                iou = compute_iou(self.segmentation_mask, self.ground_truth)
                self.iou_label.config(text=f"IoU: {iou:.4f}")
            else:
                self.iou_label.config(text="IoU: N/A (no GT)")
                
            messagebox.showinfo("Success", "Segmentation completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def save_mask(self):
        if self.segmentation_mask is None:
            messagebox.showwarning("Error", "No segmentation to save!")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")])
        if filepath:
            cv2.imwrite(filepath, self.segmentation_mask * 255)
            messagebox.showinfo("Success", f"Mask saved to:\n{filepath}")
            
    def run(self):
        self.root.mainloop()


def main():
    gui = InteractiveSegmentationGUI()
    gui.run()


if __name__ == "__main__":
    main()