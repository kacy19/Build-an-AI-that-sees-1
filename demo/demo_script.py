import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.data_preprocessing import load_and_preprocess_data

# Load model
model_path = '../models/basic_cnn_model.h5'  # Change if using a different model
model = load_model(model_path)

# Function to load and preprocess the test images
def load_test_images(test_images_folder='demo/test_images/'):
    images = []
    filenames = os.listdir(test_images_folder)
    
    for filename in filenames:
        img_path = os.path.join(test_images_folder, filename)
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        images.append(img_array)
    
    return np.vstack(images), filenames

# Load test images from the demo folder
test_images, filenames = load_test_images()

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Display results
def plot_predictions(test_images, predicted_labels, filenames):
    fig, axes = plt.subplots(1, len(test_images), figsize=(15, 5))
    
    if len(test_images) == 1:
        axes = [axes]  # Ensure axes is iterable
    
    for ax, img, label, filename in zip(axes, test_images, predicted_labels, filenames):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'{filename} - Predicted: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Plot and display the predictions
plot_predictions(test_images, predicted_labels, filenames)

# Optionally, save the result as a file (e.g., sample_predictions.png)
output_path = 'results/sample_predictions.png'
plot_predictions(test_images, predicted_labels, filenames)
plt.savefig(output_path)

print(f'Predictions saved to {output_path}')
