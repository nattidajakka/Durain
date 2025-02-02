from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the main folder containing subfolders
main_folder = '/content/drive/MyDrive/Durian'
subfolders = ['AlgalLeafSpot', 'LeafBlight', 'Leaf-Spot', 'NoDisease']

# Load images and labels
images = []
labels = []
label_map = {name: i for i, name in enumerate(subfolders)}

for subfolder in subfolders:
    path = os.path.join(main_folder, subfolder)
    for filename in os.listdir(path):
        if filename.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))  # Resize to a fixed size
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label_map[subfolder])

# Convert to numpy arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

