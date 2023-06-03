import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the saved AI model
model = tf.keras.models.load_model('saved_model')

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Process the uploaded image
        image = Image.open(file_path)
        # Preprocess the image as needed for your model
        processed_image = preprocess_image(image)

        # Convert the processed image to a format compatible with the model
        input_data = np.expand_dims(processed_image, axis=0)

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Get the predicted class label
        predicted_class = get_predicted_class(predictions)

        # Print the predicted class
        print("Predicted class:", predicted_class)

#
# Function to preprocess the image
def preprocess_image(image):
    # Preprocess the image as needed (e.g., resize, normalization, etc.)
    # Resize the image to (28, 28) pixels
    processed_image = image.resize((28, 28))
    # Convert the image to grayscale
    processed_image = processed_image.convert('L')
    # Convert the image to a NumPy array
    processed_image = np.array(processed_image)
    # Reshape the image to add the channel dimension
    processed_image = processed_image.reshape((28, 28, 1))
    # Normalize the image
    processed_image = processed_image / 255.0
    return processed_image

# Function to get the predicted class
def get_predicted_class(predictions):
    # Get the predicted class label from the predictions
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Create the main application window
window = tk.Tk()
window.title("AI Model Application")

# Create a button for image upload
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# Run the application
window.mainloop()
