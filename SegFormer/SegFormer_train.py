import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import matplotlib.pyplot as plt
from unet_dataset import XRayDataset, get_transform
from scipy.ndimage import label

# Hyperparameter setting
EPOCHS = 10
LEARNING_RATE = 1e-6
BATCH_SIZE = 40

# Initialize the feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# data preparation
# Reminder: You need to change the following file path to the true path in your computer.
transform = get_transform()
train_dataset = XRayDataset(image_dir='bob/Downloads/PKU_summerschool/data/1000_external_data/image',
                            mask_dir='bob/Downloads/PKU_summerschool/data/1000_external_data/mask',
                            feature_extractor=feature_extractor,
                            transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# model initialization
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(pred, mask):
    '''
    Calculate accuracy
    accuracy = (intersection of predicted and ground truth mask)/ (true mask)
    '''
    pred = (pred > 0.5).float()
    acc = torch.sum((mask == 1) & (pred == 1)).float() / torch.sum(mask == 1).float()
    return acc.item()

def keep_largest_regions(mask, n_largest=2):
    '''
    Defined function to only retain the two largest connected white areas in the mask (i.e., humans have two lung areas)
    '''
    # Ensure the mask is 2D format
    if mask.ndim > 2:
        mask = mask.squeeze()
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    unique, counts = np.unique(labeled, return_counts=True)

    # Find the indices (location) of the largest components
    largest_components = unique[np.argsort(-counts)][:n_largest + 1]  # +1 because background is labeled as 0

    # only remain the top 2 largest components and remove others
    filtered_mask = np.isin(labeled, largest_components[1:])  # Exclude background with index 0
    return filtered_mask.astype(np.float)

# save training accuracies and losses
train_accuracies = []
train_losses = []

# Training model
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(pixel_values=images).logits

        output_prob = torch.sigmoid(output)
        output_binary = (output_prob > 0.5).float()
        output_binary_np = output_binary.cpu().detach().numpy()

        # apply keep_largest_regions
        output_processed = np.array([keep_largest_regions(img, n_largest=2) for img in output_binary_np])
        output_processed = np.round(output_processed)  # Make sure the output is a binary image
        output_processed_torch = torch.from_numpy(output_processed).to(device)

        # calculate training loss and accuracy
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
        # calculate the average loss and accuracy
        train_loss += loss.item() * images.size(0)
        train_acc += calculate_accuracy(output_processed_torch, masks)

    train_loss_avg = train_loss / len(train_loader.dataset)
    train_acc_avg = train_acc / len(train_loader.dataset)
    train_losses.append(train_loss_avg)
    train_accuracies.append(train_acc_avg)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss_avg:.4f}, Training Acc: {train_acc_avg:.4f}')

# save the model
torch.save(model.state_dict(), 'segformer_model.pth')
