import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import numpy as np
import torch
from model import NeuralNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 1000
num_classes = 10  # 0-9
model = NeuralNet(input_size, hidden_size, num_classes)

# Load the model
FILE = "model.pth"
model.load_state_dict(torch.load(FILE, weights_only=True))
model.to(device)
model.eval()

# Function to clear the canvas


def clear_canvas():
    canvas.delete("all")

# Function to predict the digit


def classify_digit():
    # Save the canvas content as a PostScript file
    canvas.postscript(file="canvas.ps", colormode="color")

    # Convert the PostScript file to an image
    img = Image.open("canvas.ps").convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.reshape(1, -1)  # Flatten the image

    # Convert to tensor and classify
    tensor = torch.tensor(img, dtype=torch.float32).to(device)
    output = model(tensor)
    _, predicted = torch.max(output.data, 1)
    prediction_label.config(text=f"Predicted Digit: {predicted.item()}")


# Set up the GUI
root = tk.Tk()
root.title("Digit Classifier")

canvas = tk.Canvas(root, width=140, height=140, bg="white")
canvas.pack()

# Buttons for actions
btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(side="left")

btn_classify = tk.Button(root, text="Classify", command=classify_digit)
btn_classify.pack(side="right")

# Display prediction
prediction_label = tk.Label(root, text="Predicted Digit: ")
prediction_label.pack()

# Canvas drawing functionality


def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)


canvas.bind("<B1-Motion>", paint)

root.mainloop()
