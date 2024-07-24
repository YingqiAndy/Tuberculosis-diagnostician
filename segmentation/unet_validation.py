import torch
import numpy as np
from torch.utils.data import DataLoader
from unet_model import UNet
import torch.nn as nn
from unet_dataset import XRayDataset, get_transform
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Load model
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
#Reminder: You need to change the following file path to the true path in your computer.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('unet_model.pth'))
model.to(device)

# hyperparameter setting
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 40

# data load
#Reminder: You need to change the following file path to the true path in your computer.
transform = get_transform()
val_dataset = XRayDataset(image_dir='../data/200_val_external/image',
                          mask_dir='../data/200_val_external/mask',
                          transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# loss function
criterion = nn.BCEWithLogitsLoss()


def keep_largest_regions(mask, n_largest=2):
    '''
    Defined function to only retain the two largest connected white areas in the mask (i.e., the two lung areas)
    '''
    # Ensure the mask is 2D format
    if mask.ndim > 2:
        mask = mask.squeeze()
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    unique, counts = np.unique(labeled, return_counts=True)

    # Find the indices (location) of the largest components
    largest_components = unique[np.argsort(-counts)][:n_largest + 1]

    # only remain the top 2 largest components and remove others
    filtered_mask = np.isin(labeled, largest_components[1:])  # 排除背景
    return filtered_mask.astype(np.float)


def calculate_accuracy(pred, mask):
    '''
    Calculate accuracy
    accuracy = (intersection of predicted and ground truth mask)/ (true mask)
    '''
    # Check if the mask of the predicted output is only 0 and 1 (binary image or not),
    # if not, round it
    if not torch.all((pred == 0) | (pred == 1)):
        print("Warning: Predictions contain values other than 0 and 1")
        pred = torch.round(pred)

    acc = torch.sum((mask == 1) & (pred > 0.5)) / torch.sum(mask == 1).float()
    return acc.item()


# validation process
val_losses = []
val_accuracies = []  # Stores the accuracy for each epoch
model.eval()

for epoch in range(EPOCHS):
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss = criterion(output, masks)

            # Use sigmoid() to convert the raw output into probability values,
            # and then converted them into binary format (0 or 1) based on a threshold of 0.5
            output_prob = torch.sigmoid(output)
            output_binary = (output_prob > 0.5).float()
            output_binary_np = output_binary.cpu().detach().numpy()

            # apply keep_largest_regions
            output_processed = np.array([keep_largest_regions(img, n_largest=2) for img in output_binary_np])
            output_processed = np.round(output_processed)  # Make sure the output is a binary image
            output_processed_torch = torch.from_numpy(output_processed).to(device)

            val_loss += loss.item() * images.size(0)
            val_accuracy += calculate_accuracy(output_processed_torch, masks)

    # Calculate the average loss and average accuracy for the current epoch
    val_loss_avg = val_loss / len(val_loader.dataset)
    val_accuracy_avg = val_accuracy / len(val_loader.dataset)

    # add result in current epoch to val_losses and val_accuracies
    val_losses.append(val_loss_avg)
    val_accuracies.append(val_accuracy_avg)

    print(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy_avg:.4f}')

print(val_losses)
print(val_accuracies)
