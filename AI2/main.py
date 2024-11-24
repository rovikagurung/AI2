import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Function to load images from folders
def load_images_from_folder(folder, label_index):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (48, 48))  # Resize to the target size (48x48)
            images.append(img_resized)
            labels.append(label_index)
        else:
            print(f"Skipping invalid image: {filename}")
    return images, labels

# Load the dataset
def load_dataset(base_folder):
    images = []
    labels = []
    classes = os.listdir(base_folder)  # Get class names from subfolders
    print(f"Classes found: {classes}")
    for i, class_name in enumerate(classes):
        folder_path = os.path.join(base_folder, class_name)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            imgs, lbls = load_images_from_folder(folder_path, i)
            images.extend(imgs)
            labels.extend(lbls)
    images = np.array(images).reshape(-1, 48, 48, 1) / 255.0  # Normalize images and reshape
    labels = to_categorical(labels, len(classes))  # One-hot encode labels
    return images, labels, classes

# Build the CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to predict emotion for a new image
def predict_emotion(image_path, model, class_names):
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))  # Load image
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)  # Get the index of the predicted class
    return class_names[emotion_index], predictions[0]  # Return class and confidence scores

# Function to test the model on a new image
def test_new_image(image_path, model, class_names):
    emotion, confidence_scores = predict_emotion(image_path, model, class_names)
    print(f"Predicted Emotion: {emotion}")
    print(f"Confidence Scores: {confidence_scores}")

# Main script
if __name__ == "__main__":
    base_folder = "./dataset"  # Path to dataset folder
    model_path = "emotion_model.h5"  # Model save path
    new_image_path = "path_to_your_test_image.jpg"  # Path to the test image

    # Load dataset
    print("Loading dataset...")
    images, labels, class_names = load_dataset(base_folder)
    print(f"Loaded {len(images)} images across {len(class_names)} classes: {class_names}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train[0].shape, len(class_names))
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Train the model

    # Evaluate model on test data
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    model.save(model_path)
    print(f"Model saved as '{model_path}'.")

    # Test a new image
    if os.path.exists(new_image_path):
        print(f"Testing on new image: {new_image_path}")
        test_new_image(new_image_path, model, class_names)
    else:
        print(f"New image not found: {new_image_path}")
