import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from UnetModel import UNet  # Import the original UNet model
from preprocessing import X_train, X_val, y_train, y_val

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the U-Net model with 3 input channels
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
