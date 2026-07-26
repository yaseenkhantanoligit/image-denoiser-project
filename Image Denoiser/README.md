# AI Image Denoiser — Lite

No training. No TensorFlow. No PyQt5. Just one file.

## Run it
```
pip install -r requirements.txt
python main.py
```
(Tkinter ships with Python on Windows/Mac. On Linux, if you get a
"No module named tkinter" error, run: `sudo apt install python3-tk`)

A window opens. Click **Open Image**, pick a **method** and **strength**,
click **Denoise**, then **Save Result**.

## Methods
- **Non-Local Means** — best quality, a little slower
- **Bilateral** — smooths noise but keeps edges sharp
- **Median** — best for salt & pepper speckles
- **Gaussian** — fastest, general blur-based smoothing

These are classic, well-tested OpenCV algorithms — no model files, no GPU,
no waiting for training to finish. It just works the moment you open an image.
