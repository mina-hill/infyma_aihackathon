import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Define paths (Update as needed)
test_dir = "/kaggle/input/diabetic-retinopathy-balanced/content/Diabetic_Balanced_Data/test"  # Modify based on your dataset structure
model_path = "/kaggle/working/diabetic_retinopathy_densenet121_finetuned.h5"

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Function to randomly select one image from each class
def get_random_images(test_dir):
    class_images = {}  # Dictionary to store class-wise random images

    for class_name in os.listdir(test_dir):  # Iterate over class folders
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            img_name = random.choice(os.listdir(class_path))  # Pick a random image
            img_path = os.path.join(class_path, img_name)
            class_images[class_name] = img_path

    return class_images

# Function to preprocess an image for model prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match DenseNet input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict, display images, and compute accuracy
def predict_and_show(class_images):
    correct_predictions = 0
    total_images = len(class_images)

    for class_name, img_path in class_images.items():
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)  # Get predicted class index
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Convert class name to index (assuming class names are numeric like '0', '1', '2', etc.)
        actual_class_index = int(class_name)

        # Check if prediction is correct
        is_correct = predicted_class_index == actual_class_index
        if is_correct:
            correct_predictions += 1

        # Show the image with prediction
        plt.figure(figsize=(4, 4))
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Actual: {actual_class_index} | Predicted: {predicted_class_index}\nConfidence: {confidence:.2f}%")
        plt.show()

        # Print accuracy for this image
        print(f"Image: {img_path}\nActual Class: {actual_class_index} | Predicted Class: {predicted_class_index}")
        print(f"Confidence: {confidence:.2f}% | Correct: {is_correct}\n")

    # Calculate overall accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")

# Run the test
class_images = get_random_images(test_dir)
predict_and_show(class_images)
