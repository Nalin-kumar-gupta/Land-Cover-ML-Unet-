import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# Create lists to store images and masks
images = []
masks = []

# Define a function to read and preprocess images and masks
def preprocess_data(image_path, mask_path):
    # Load and resize the image (you can adjust the size)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (256, 256))  # Resize to the desired input size
    image = image / 255.0  # Normalize pixel values (0-1 range)

    # Load and resize the mask (binary mask)
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (256, 256))  # Resize to the same size as the image
    mask = np.where(mask > 128, 1, 0)  # Binarize the mask at threshold 128

    return image, mask


data_dir = r"E:\vscode\Land Cover Classification ML\Project\dataset\train"

# Iterate through the dataset directory
for filename in os.listdir(data_dir):
    if filename.endswith("_sat.jpg"):
        image_path = os.path.join(data_dir, filename)
        mask_filename = filename.replace("_sat.jpg", "_mask.png")
        mask_path = os.path.join(data_dir, mask_filename)

        if os.path.isfile(mask_path):
            # Load and preprocess data
            image, mask = preprocess_data(image_path, mask_path)

            # Append to the lists
            images.append(image)
            masks.append(mask)

# Convert lists to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Print the shape of the arrays (for verification)
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)





images = torch.from_numpy(images).float()
masks = torch.from_numpy(masks).long()  # Use 'long' data type for classification
X_train = images
y_train = masks





data_dir = r"E:\vscode\Land Cover Classification ML\Project\dataset\valid"



# Iterate through the dataset directory
for filename in os.listdir(data_dir):
    if filename.endswith("_sat.jpg"):
        image_path = os.path.join(data_dir, filename)
        mask_filename = filename.replace("_sat.jpg", "_mask.png")
        mask_path = os.path.join(data_dir, mask_filename)

        if os.path.isfile(mask_path):
            # Load and preprocess data
            image, mask = preprocess_data(image_path, mask_path)

            # Append to the lists
            images.append(image)
            masks.append(mask)

# Convert lists to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Print the shape of the arrays (for verification)
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)




images = torch.from_numpy(images).float()
masks = torch.from_numpy(masks).long()  # Use 'long' data type for classification
X_val = images
y_val = masks


