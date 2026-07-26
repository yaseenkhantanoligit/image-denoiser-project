"""
AI Image Denoiser - Lite Edition
=================================
ONE file. ONE click. No training, no TensorFlow, no GPU, no dataset.

HOW TO RUN:
    1) pip install -r requirements.txt
    2) python main.py

That's it. A window opens -> Open Image -> pick a method -> Denoise -> Save.

Uses OpenCV's built-in, pretrained-free denoising algorithms:
    - Non-Local Means (best quality, a bit slower)
    - Bilateral Filter (edge-preserving, fast)
    - Median Blur (great for salt & pepper noise)
    - Gaussian Blur (fast, general smoothing)
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

METHODS = ["Non-Local Means (best quality)", "Bilateral (edge-preserving)",
           "Median (salt & pepper)", "Gaussian (fast)"]


def denoise(img_bgr: np.ndarray, method: str, strength: int) -> np.ndarray:
    if method == METHODS[0]:
        h = max(3, strength)
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
    if method == METHODS[1]:
        d = max(3, strength // 3)
        return cv2.bilateralFilter(img_bgr, d, strength * 2, strength * 2)
    if method == METHODS[2]:
        k = strength if strength % 2 == 1 else strength + 1
        k = max(3, min(k, 15))
        return cv2.medianBlur(img_bgr, k)
    if method == METHODS[3]:
        k = strength if strength % 2 == 1 else strength + 1
        k = max(3, min(k, 25))
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    return img_bgr


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Denoiser - Lite")
        self.geometry("900x560")
        self.minsize(760, 480)

        self.original_bgr = None
        self.denoised_bgr = None
        self.image_path = None
        self._tk_before = None
        self._tk_after = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="1. Open Image", command=self.open_image).pack(side="left", padx=4)

        ttk.Label(top, text="Method:").pack(side="left", padx=(16, 4))
        self.method_var = tk.StringVar(value=METHODS[0])
        ttk.Combobox(top, textvariable=self.method_var, values=METHODS,
                     state="readonly", width=26).pack(side="left")

        ttk.Label(top, text="Strength:").pack(side="left", padx=(16, 4))
        self.strength_var = tk.IntVar(value=10)
        ttk.Scale(top, from_=1, to=25, variable=self.strength_var,
                  orient="horizontal", length=140).pack(side="left")

        self.denoise_btn = ttk.Button(top, text="2. Denoise", command=self.run_denoise, state="disabled")
        self.denoise_btn.pack(side="left", padx=(16, 4))

        self.save_btn = ttk.Button(top, text="3. Save Result", command=self.save_result, state="disabled")
        self.save_btn.pack(side="left", padx=4)

        images = ttk.Frame(self)
        images.pack(fill="both", expand=True, padx=8, pady=8)
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)
        images.rowconfigure(1, weight=1)

        ttk.Label(images, text="Original", anchor="center").grid(row=0, column=0, sticky="ew")
        ttk.Label(images, text="Denoised", anchor="center").grid(row=0, column=1, sticky="ew")

        self.before_label = tk.Label(images, bg="#222", fg="#aaa", text="No image loaded")
        self.before_label.grid(row=1, column=0, sticky="nsew", padx=4)
        self.after_label = tk.Label(images, bg="#222", fg="#aaa", text="Not denoised yet")
        self.after_label.grid(row=1, column=1, sticky="nsew", padx=4)

        self.status = tk.StringVar(value="Start by opening an image.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

        images.bind("<Configure>", lambda e: self._refresh_previews())

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]
        )
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return

        self.image_path = path
        self.original_bgr = img
        self.denoised_bgr = None
        self.save_btn.config(state="disabled")
        self.denoise_btn.config(state="normal")
        self.after_label.config(image="", text="Not denoised yet")
        self.status.set(f"Loaded: {os.path.basename(path)}  ({img.shape[1]}x{img.shape[0]})")
        self._refresh_previews()

    def run_denoise(self):
        if self.original_bgr is None:
            return
        self.status.set("Denoising...")
        self.update_idletasks()
        start = time.perf_counter()
        try:
            self.denoised_bgr = denoise(self.original_bgr, self.method_var.get(), int(self.strength_var.get()))
        except Exception as exc:
            messagebox.showerror("Denoising failed", str(exc))
            self.status.set("Denoising failed.")
            return
        elapsed = (time.perf_counter() - start) * 1000
        self.save_btn.config(state="normal")
        self.status.set(f"Done in {elapsed:.0f} ms using {self.method_var.get()}.")
        self._refresh_previews()

    def save_result(self):
        if self.denoised_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="denoised_output.png",
        )
        if not path:
            return
        cv2.imencode(os.path.splitext(path)[1], self.denoised_bgr)[1].tofile(path)
        self.status.set(f"Saved: {path}")

    def _to_photo(self, bgr, max_size):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(max_size)
        return ImageTk.PhotoImage(pil)

    def _refresh_previews(self):
        w = max(100, self.before_label.winfo_width() - 8)
        h = max(100, self.before_label.winfo_height() - 8)
        if self.original_bgr is not None:
            self._tk_before = self._to_photo(self.original_bgr, (w, h))
            self.before_label.config(image=self._tk_before, text="")
        if self.denoised_bgr is not None:
            self._tk_after = self._to_photo(self.denoised_bgr, (w, h))
            self.after_label.config(image=self._tk_after, text="")


if __name__ == "__main__":
    App().mainloop()
