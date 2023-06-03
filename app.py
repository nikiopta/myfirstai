import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the saved AI model
model = tf.keras.models.load_model('saved_model')

# Load the class labels
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Create the main application window
window = tk.Tk()
window.title("AI Model Application")

# Create a global label for displaying the predicted class
prediction_label = tk.Label(window, text="", font=("Helvetica", 16))
prediction_label.pack()

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Process the uploaded image
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize the image for display

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(processed_image)

        # Get the predicted class label
        predicted_class = get_predicted_class(predictions)

        # Display the uploaded image and the predicted class
        display_image(image)
        display_prediction(predicted_class)

def display_image(image):
    # Create a new window to display the image
    image_window = tk.Toplevel(window)
    image_window.title("Uploaded Image")

    # Convert the image to a Tkinter-compatible format
    img_tk = ImageTk.PhotoImage(image)

    # Create a label to display the image
    image_label = tk.Label(image_window, image=img_tk)
    image_label.image = img_tk
    image_label.pack()

def display_prediction(predicted_class):
    # Update the predicted class label
    prediction_label.config(text="Predicted class: " + class_labels[predicted_class])

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to (28, 28) pixels
    processed_image = image.resize((28, 28))
    # Convert the image to grayscale
    processed_image = processed_image.convert('L')
    # Convert the image to a NumPy array
    processed_image = np.array(processed_image)
    # Reshape the image to add the channel dimension
    processed_image = processed_image.reshape((1, 28, 28, 1))
    # Normalize the image
    processed_image = processed_image / 255.0
    return processed_image

# Function to get the predicted class
def get_predicted_class(predictions):
    # Get the predicted class label from the predictions
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Create a button for image upload
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# Run the application
window.mainloop()
