import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tkinter import LEFT, BOTTOM, RIGHT, TOP

import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras import backend as K
import os

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

# ============================================
# Load model
# ============================================
model = tf.keras.models.load_model(
    "lane_model.h5",
    custom_objects={"dice_loss": dice_loss},
    compile=False
)

def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb / 255.0
    return np.expand_dims(frame_norm, axis=0)

def postprocess_output(original_frame, prediction, threshold=0.5):
    mask = prediction[0, :, :, 0]
    mask = (mask > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_frame.shape[1], original_frame.shape[0]))
    overlay = np.zeros_like(original_frame)
    overlay[:, :, 1] = mask
    combined = cv2.addWeighted(original_frame, 0.8, overlay, 0.5, 0)
    return combined

cap = None
processing = False

def browse_file():
    global cap, processing
    if processing:
        return
    file_path = filedialog.askopenfilename(
        title="Select Image or Video",
        filetypes=[
            ("All Supported", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi"),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi")
        ]
    )
    if not file_path:
        return
    stop_video()

    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(file_path)
    elif ext in [".mp4", ".avi"]:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            status_label.configure(text="Error: Could not open video.", text_color="red")
            return
        processing = True
        status_label.configure(text="Playing video... Click Stop to halt.", text_color="#00ffcc")
        process_video()

def process_image(path):
    global processing
    processing = True
    status_label.configure(text="Processing image...", text_color="#ffcc00")
    root.update()

    image = cv2.imread(path)
    if image is None:
        status_label.configure(text="Error loading image.", text_color="red")
        processing = False
        return

    processed = preprocess_frame(image)
    prediction = model.predict(processed, verbose=0)
    output = postprocess_output(image, prediction)

    display_images(image, output)
    status_label.configure(text="Image processed successfully!", text_color="#55ff99")
    processing = False

def process_video():
    global cap, processing
    if not processing or not cap or not cap.isOpened():
        return
    ret, frame = cap.read()
    if not ret:
        stop_video()
        status_label.configure(text="Video ended.", text_color="#8888ff")
        return
    processed = preprocess_frame(frame)
    prediction = model.predict(processed, verbose=0)
    output = postprocess_output(frame, prediction)
    display_images(frame, output)
    root.after(33, process_video)  

def stop_video():
    global cap, processing
    processing = False
    if cap:
        cap.release()
        cap = None
    status_label.configure(text="Ready. Select a file to begin.", text_color="white")

def display_images(original, output):
    display_size = (420, 340)
    orig_resized = cv2.resize(original, display_size)
    out_resized = cv2.resize(output, display_size)
    orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB)

    img1 = ImageTk.PhotoImage(Image.fromarray(orig_rgb))
    img2 = ImageTk.PhotoImage(Image.fromarray(out_rgb))

    original_label.configure(image=img1)
    original_label.image = img1
    output_label.configure(image=img2)
    output_label.image = img2

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.title("AI Lane Detection System")
root.geometry("1000x660")
root.resizable(False, False)


title = ctk.CTkLabel(
    root,
    text="AI Lane Detection System",
    font=("Poppins", 28, "bold"),
    text_color="#00ffaa"
)
title.pack(pady=(25, 10))

content_frame = ctk.CTkFrame(root, fg_color="#1b1e24", corner_radius=12)
content_frame.pack(pady=30)

left_frame = ctk.CTkFrame(content_frame, fg_color="#1b1e24", corner_radius=12)
right_frame = ctk.CTkFrame(content_frame, fg_color="#1b1e24", corner_radius=12)
left_frame.grid(row=0, column=0, padx=25)
right_frame.grid(row=0, column=1, padx=25)

ctk.CTkLabel(left_frame, text="Original", font=("Segoe UI", 14, "bold")).pack(pady=10)
ctk.CTkLabel(right_frame, text="Detected Lanes", font=("Segoe UI", 14, "bold"), text_color="#00ffaa").pack(pady=10)

original_label = ctk.CTkLabel(left_frame, width=420, height=340, text="")
original_label.pack(padx=10, pady=10)
output_label = ctk.CTkLabel(right_frame, width=420, height=340, text="")
output_label.pack(padx=10, pady=10)

control_frame = ctk.CTkFrame(root, fg_color="#111317", corner_radius=12)
control_frame.pack(pady=25)

browse_btn = ctk.CTkButton(control_frame, text="📂  Browse File", command=browse_file, width=150, height=45)
browse_btn.pack(side=LEFT, padx=25, pady=15)

stop_btn = ctk.CTkButton(control_frame, text="⏹️  Stop Video", command=stop_video, fg_color="#ff5555", hover_color="#ff6666", width=150, height=45)
stop_btn.pack(side=LEFT, padx=25, pady=15)

quit_btn = ctk.CTkButton(control_frame, text="❌  Quit", command=root.destroy, fg_color="#333333", hover_color="#555555", width=150, height=45)
quit_btn.pack(side=RIGHT, padx=25, pady=15)


status_label = ctk.CTkLabel(root, text="Ready. Select a file to begin.", font=("Consolas", 12))
status_label.pack(fill="x", side="bottom", pady=(0, 10))


placeholder = np.zeros((340, 420, 3), dtype=np.uint8)

placeholder_text = cv2.putText(
    placeholder.copy(), "No File Selected", (100, 180),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 3
)
ph1 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(placeholder_text, cv2.COLOR_BGR2RGB)))
ph2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)))
original_label.configure(image=ph1)
original_label.image = ph1
output_label.configure(image=ph2)
output_label.image = ph2


root.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()