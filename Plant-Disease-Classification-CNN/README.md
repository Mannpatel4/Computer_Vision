# Plant Disease Classification (Custom CNN)

[![Plant Disease Demo](https://img.shields.io/badge/demo-plant--disease-brightgreen)](https://https://plant-disease-classifiers.streamlit.app)

##  Project Overview
This project focuses on classifying images of plant leaves into healthy or diseased categories. The primary goal was to build a Convolutional Neural Network (CNN) completely from scratch to deeply understand image preprocessing, feature extraction, and model architecture design.

##  Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** `ImageDataGenerator`

##  Approach
Instead of using pre-trained weights, I designed a custom sequential CNN:
1. **Convolutional Layers:** Multiple `Conv2D` layers paired with `MaxPooling2D` to extract spatial features (edges, spots, leaf textures) while reducing image dimensionality.
2. **Data Augmentation:** Applied real-time transformations (rescaling, zooming, flipping) to artificially expand the dataset and make the model robust against different lighting and angles.
3. **Regularization:** Integrated `Dropout` layers to randomly turn off neurons during training, forcing the network to learn generalized patterns rather than memorizing the training data.

##  Results
* Successfully built a functional data pipeline capable of handling raw image directories.
* Achieved strong baseline accuracy, demonstrating a fundamental understanding of how raw pixels are mathematically transformed into categorical predictions.