# Importing necessary libraries
import tkinter as tk  # Library for creating GUI applications
from tkinter import filedialog  # Module that provides dialog boxes for file selection
from tkinter import ttk  # Module for creating themed widgets in tkinter

import numpy as np  # Library for numerical operations
import cv2  # Library for computer vision tasks
from PIL import Image, ImageTk  # Library for manipulating images

# Importing deep learning libraries
import tensorflow as tf  # Open-source library for machine learning
from tensorflow import keras  # High-level API for building and training deep learning models

# Importing specific deep learning models from Keras
from keras.applications.efficientnet_v2 import EfficientNetV2M  # EfficientNetV2M model
from keras.applications.mobilenet_v2 import MobileNetV2  # MobileNetV2 model
from keras.applications.xception import Xception  # Xception model
from keras.applications.resnet_v2 import ResNet50V2  # ResNet50V2 model
from keras.applications.inception_v3 import InceptionV3  # InceptionV3 model
from keras.applications.densenet import DenseNet121  # DenseNet121 model
from keras.applications.nasnet import NASNetMobile  # NASNetMobile model

# Create the main application window
app = tk.Tk()
app.title("Image Prediction App")  # Set the title of the application

# Function to browse and select an image
def browse_image():
    global image, photo  # Declare selected_model and model as global variables
    
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Open and resize the image
        image = Image.open(file_path)
        image = image.resize((480, 480))
        
        # Display the resized image on the GUI
        photo = ImageTk.PhotoImage(image=image)
        image_label.config(image=photo)
        image_label.image = photo

# Function to predict the content of the selected image
def predict_image():
    global image, selected_model, model, selected_model, model, selected_image
    
    # Get the selected model from the combobox
    selected_model = model_combobox.get()

    # Define the selected model based on the choice and perform preprocessing accordingly
    if selected_model == "EfficientNetV2":
        # Convert to numpy array and perform additional preprocessing
        selected_image = np.array(image)
        selected_image = cv2.resize(selected_image, (480, 480))  # Resize to (480, 480)
        selected_image = np.expand_dims(selected_image, axis=0)  # Add a batch dimension
    elif selected_model == "Xception" or selected_model == "InceptionV3":
        # Convert to numpy array and perform additional preprocessing
        selected_image = np.array(image)
        selected_image = cv2.resize(selected_image, (299, 299))  # Resize to (299, 299)
        selected_image = np.expand_dims(selected_image, axis=0)  # Add a batch dimension
    else:
        # Convert to numpy array and perform additional preprocessing
        selected_image = np.array(image)
        selected_image = cv2.resize(selected_image, (224, 224))  # Resize to (224, 224)
        selected_image = np.expand_dims(selected_image, axis=0)  # Add a batch dimension
    
    predictions = None  # Initialize predictions with a default value

    # Perform predictions based on the selected model
    if selected_model == "EfficientNetV2":
        model = EfficientNetV2M()
        predictions = model.predict(selected_image)
    elif selected_model == "MobileNetV2":
        model = MobileNetV2()
        predictions = model.predict(selected_image)
    elif selected_model == "Xception":
        model = Xception()
        predictions = model.predict(selected_image)
    elif selected_model == "InceptionV3":
        model = InceptionV3()
        predictions = model.predict(selected_image)
    elif selected_model == "ResNet50V2":
        model = ResNet50V2()
        predictions = model.predict(selected_image)
    elif selected_model == "DenseNet121":
        model = DenseNet121()
        predictions = model.predict(selected_image)
    elif selected_model == "NASNetMobile":
        model = NASNetMobile()
        predictions = model.predict(selected_image)
    # Decode the predictions and extract the predicted label
    predicted_label = keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0][1]
    prediction_label.config(text=f"Predicted: {predicted_label}")

# Insert Image Button
insert_button = tk.Button(app, text="Insert Image", command=browse_image)
insert_button.grid(row=0, column=0)

# Display Inserted Image
image_label = tk.Label(app)
image_label.grid(row=0, column=1)

# Model Selection Combobox
model_label = tk.Label(app, text="Select Model:")
model_label.grid(row=1, column=0)
models = ["EfficientNetV2", "MobileNetV2", "Xception", "ResNet50V2", "InceptionV3", "DenseNet121", "NASNetMobile"]
model_combobox = ttk.Combobox(app, values=models)
model_combobox.current(0)  # Set the default model as the first one in the list
model_combobox.grid(row=1, column=1)

# Predict Button
predict_button = tk.Button(app, text="Predict", command=predict_image)
predict_button.grid(row=2, column=0, columnspan=2)

# Display Predicted Item
prediction_label = tk.Label(app, text="Predicted: ")
prediction_label.grid(row=3, column=0, columnspan=2)

# Run the application main loop
app.mainloop()