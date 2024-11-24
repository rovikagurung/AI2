import tkinter as tk
from tkinter import filedialog, messagebox
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk, ImageEnhance

# Load the pre-trained model
try:
    model = load_model('emotion_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']  # Example class labels

def predict_emotion(image_path):
    if model is None:
        return None, 0, None
    
    try:
        # Load image in grayscale for prediction
        img = load_img(image_path, target_size=(48, 48), color_mode="grayscale")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        predictions = model.predict(img_array)
        
        # Print the prediction results (confidence scores for all classes)
        print(f"Prediction Scores: {predictions}")
        
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        # Load the image in RGB for display
        img_rgb = load_img(image_path, target_size=(400, 400))  # Resize to fit in the window
        img_rgb = img_rgb.convert("RGB")  # Convert to RGB if needed
        
        return predicted_class, confidence, img_rgb
    except Exception as e:
        messagebox.showerror("Error", f"Error processing the image: {e}")
        return None, 0, None

def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    if file_path:
        emotion, confidence, img = predict_emotion(file_path)
        if emotion:
            # Update emotion and confidence text
            emotion_label.config(text=f"Emotion: {emotion}\nConfidence: {confidence*100:.2f}%")
            
            # Resize and enhance image
            img = resize_image(img)  # Resize for display
            img = enhance_image(img)  # Enhance sharpness
            
            # Convert the image to Tkinter-compatible format and display it
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk  # Keep a reference to the image to avoid garbage collection
        else:
            messagebox.showerror("Error", "Could not process the image.")

# Resize the image while keeping the aspect ratio
def resize_image(img, base_width=400):
    width, height = img.size
    ratio = height / width
    new_width = base_width
    new_height = int(base_width * ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resampling
    return img

# Enhance the image for better clarity
def enhance_image(img):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2
    return img

# Set up the Tkinter window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("500x500")  # Adjust the window size

# Create and place the button to open the file dialog
button = tk.Button(root, text="Open Image", command=open_image)
button.pack(pady=20)

# Label to display the emotion and confidence
emotion_label = tk.Label(root, text="Emotion: \nConfidence: ", font=("Helvetica", 14))
emotion_label.pack(pady=20)

# Label to display the image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
