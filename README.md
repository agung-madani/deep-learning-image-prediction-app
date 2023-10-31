# Deep Learning Image Prediction Application

This repository contains a simple Python script that enables users to predict the contents of an image using various deep learning models. The application is built using the Tkinter library for creating a graphical user interface (GUI) and leverages the power of deep learning models from the Keras library for image classification.

 Beagle | Bee
------------- | -------------
![Screenshot 2023-10-31 091224](https://github.com/agung-madani/image-prediction-app-tkinter/assets/121701309/b1fd8786-ae54-4479-9738-eb47caea8880)  | ![Screenshot 2023-10-31 085616](https://github.com/agung-madani/image-prediction-app-tkinter/assets/121701309/310b796d-3c74-4ccc-beeb-1dcd6ce0123f)

## How to Use

- Clone the repository to your local machine.
```BASH
git clone https://github.com/agung-madani/deep-learning-image-prediction-app.git
```
- Make sure you have the necessary libraries installed. You can install them using pip:
```BASH
pip install opencv-python tensorflow tkinter
```
- Run the `main.py` script to start the image prediction application.
```BASH
python main.py
```
- Click on the "Insert Image" button to select an image for prediction.
- Choose a deep learning model from the provided options in the dropdown menu.
- Click on the "Predict" button to predict the contents of the selected image.
- The predicted label will be displayed on the GUI.

## Libraries Used

- Tkinter: Library for creating GUI applications in Python.
- NumPy: Library for numerical operations in Python.
- OpenCV: Library for computer vision tasks.
- Pillow: Library for image manipulation.
- TensorFlow: Open-source library for machine learning.
- Keras: High-level API for building and training deep learning models.

## Available Deep Learning Models

- EfficientNetV2
- MobileNetV2
- Xception
- ResNet50V2
- InceptionV3
- DenseNet121
- NASNetMobile

Feel free to explore, modify, and enhance the application based on your specific requirements.
