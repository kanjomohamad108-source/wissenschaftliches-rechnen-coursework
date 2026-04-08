import numpy as np
import mnist
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import scipy.ndimage as ndi


def load_dataset(dataset = 'cg-digits'):

    if dataset == 'cg-digits':
        # Load your custom .npz file
        data = np.load("data/cg-digits.npz")
        X = data['X']          # shape: (N, 28, 28)
        t = data['y']          # labels 0-9

        # Flatten images for linear models
        N = X.shape[0]
        X = X.reshape(N, -1)  # (N, 784) 

        # Shuffle dataset to ensure all students numbers could be taken into training
        indices = np.random.permutation(N)
        X = X[indices]
        t = t[indices]

        # Split 80/20
        split = int(0.8 * N)
        X_train, X_test = X[:split], X[split:]
        t_train, t_test = t[:split], t[split:]

    elif dataset == 'mnist':
        X_train = mnist.train_images().astype(np.float64).reshape(60000, 28 * 28) / 255
        t_train = mnist.train_labels().astype(np.float64).reshape(60000,)
        X_test = mnist.test_images().astype(np.float64).reshape(10000, 28 * 28) / 255
        t_test = mnist.test_labels().astype(np.float64).reshape(10000,)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    n_tr = len(X_train)
    n_te = len(X_test)

    return np.hstack((X_train, np.ones((n_tr, 1)))), t_train, np.hstack((X_test, np.ones((n_te,1)))), t_test


### Base Classes 
class Model:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def backward(self):
        raise NotImplementedError
    
    def update(self, dw):
        self._w += dw

    def save(self, filename: str):
        modelpath = os.path.join(os.getcwd(), '/data/models')
        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)

        file = os.path.join(modelpath)
        np.save(file, self._w)


class Loss:
    def forward(self, t: np.ndarray, y: np.ndarray) -> np.float64:
        raise NotImplementedError

    def __call__(self, t: np.ndarray, y: np.ndarray) -> np.float64:
        return self.forward(t, y)
    
    def backward(self) -> np.ndarray:
        return NotImplementedError
    
    # ----- Configuration -----
CANVAS_SIZE = 150       # Canvas resolution
IMG_SIZE = 28           # Final output resolution
BRUSH_RADIUS = 8       # Brush size for drawing


class DrawGUI:
    def __init__(self, master):
        self.master = master
        master.title("Draw Digit")

        # PIL Image for drawing (high-res)
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.img)

        # Tkinter Canvas
        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()

        # Track mouse movement
        self.canvas.bind("<B1-Motion>", self.paint)
        #Label for result

        # Buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack()
        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side="left", padx=5)
        self.save_btn = tk.Button(btn_frame, text="Get 28x28 Array", command=self.get_array)
        self.save_btn.pack(side="left", padx=5)

        # For displaying the downscaled image
        self.preview_label = tk.Label(master)
        self.preview_label.pack(pady=5)
        self.pred_label = tk.Label(master, text="", font=("Arial", 20))
        self.pred_label.pack()
        

    def paint(self, event):
        x, y = event.x, event.y
        # Draw soft circle for smooth edges
        bbox = [x - BRUSH_RADIUS, y - BRUSH_RADIUS, x + BRUSH_RADIUS, y + BRUSH_RADIUS]
        self.draw.ellipse(bbox, fill=0)

        # Update Canvas for user feedback
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

    def clear(self):
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.canvas.delete("all")
        self.preview_label.config(image='')
        self.pred_label.config(text="")

    def get_array(self):
        img_gray = self.img.convert("L")
        arr = np.array(img_gray, dtype=np.uint8)

        arr = 255 - arr

        coords = np.column_stack(np.where(arr > 0))
        if coords.size == 0:
            return np.zeros((28, 28), dtype=np.float64)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        arr = arr[y_min:y_max+1, x_min:x_max+1]

        img_digit = Image.fromarray(arr, mode="L")

        w, h = img_digit.size
        scale = 20.0 / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        img_small = img_digit.resize((new_w, new_h), resample=Image.LANCZOS)

        img_28 = Image.new("L", (28, 28), color=0)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        img_28.paste(img_small, (x_offset, y_offset))

        arr = np.array(img_28, dtype=np.float64) / 255.0

        cy, cx = ndi.center_of_mass(arr)
        if not np.isnan(cx):
            shift_x = int(round(14 - cx))
            shift_y = int(round(14 - cy))
            arr = ndi.shift(arr, shift=(shift_y, shift_x), order=1)

        if hasattr(self, "callback"):
            self.callback(arr)

        preview = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        preview = preview.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
        self.tk_preview = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=self.tk_preview)

        return arr