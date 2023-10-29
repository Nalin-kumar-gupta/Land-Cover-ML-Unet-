import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Expansive path (decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.encoder(x)
        # Bottleneck
        x2 = self.bottleneck(x1)
        # Expansive path
        x3 = self.decoder(x2)
        # Final output
        out = self.out(x3)
        return out

# Set your data directories
data_dir = r'E:\vscode\Land Cover Classification ML\Project\dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'valid')

# Load and preprocess your data
def load_data(data_dir):
    image_list = []
    mask_list = []
    
    for file in os.listdir(data_dir):
        if file.endswith('_sat.jpg'):
            image_path = os.path.join(data_dir, file)
            mask_path = os.path.join(data_dir, file.replace('_sat.jpg', '_mask.png'))

            # Load and preprocess images
            image = preprocess_image(image_path)
            mask = preprocess_mask(mask_path)

            image_list.append(image)
            mask_list.append(mask)

    return image_list, mask_list

def preprocess_image(image_path):
    # Implement image preprocessing (e.g., resize, normalize)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(Image.open(image_path))
    return image

def preprocess_mask(mask_path):
    # Implement mask preprocessing (e.g., resize, binarization)
    mask = Image.open(mask_path)
    mask = (np.array(mask) > 128).astype(np.uint8)
    return mask

X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.stack(X_train)
y_train = torch.stack(y_train)
X_val = torch.stack(X_val)
y_val = torch.stack(y_val)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the U-Net model
in_channels = 3  # RGB input
out_channels = 7  # 7 classes for land cover
model = UNet(in_channels, out_channels)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for this epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # Validation
    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    # Print average validation loss for this epoch
    print(f'Validation Loss: {validation_loss / len(val_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
