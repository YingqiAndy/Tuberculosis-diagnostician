import torch
from torch.utils.data import DataLoader
import numpy as np
from unet_model import UNet
from unet_dataset import XRayDataset, get_transform
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure

# Load our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=1)
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
# Reminder: You need to change the following file path to the true path in your computer.
model.load_state_dict(torch.load('unet_model.pth'))
model.to(device)

BATCH_SIZE = 40  #set batch size
transform = get_transform()
# Reminder: You need to change the following file path to the true path in your computer.
val_dataset = XRayDataset(image_dir='../data/200_val_external/image',
                          mask_dir='../data/200_val_external/mask',
                          transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)



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
    largest_components = unique[np.argsort(-counts)][:n_largest + 1]  # +1 because background is labeled as 0

    # # only remain the top 2 largest components and remove others
    filtered_mask = np.isin(labeled, largest_components[1:])  # Exclude background with index 0
    return filtered_mask.astype(np.float)


def smooth_lung_mask(mask, structure_size=8):
    '''
    Smooth the lung mask, using binary opening and closing to remove small holes or small irregular edges in the mask,
    to make the mask more continuous and consistent
    '''
    # Create a structural element
    binary_structure = generate_binary_structure(2, structure_size)

    # Apply binary opening and binary closing to smooth the mask
    mask_opened = binary_opening(mask, structure=binary_structure)
    mask_smoothed = binary_closing(mask_opened, structure=binary_structure)

    return mask_smoothed.astype(np.float64)



def visualize_model(model, data_loader, num_examples=5):
    '''
    Show comparison of original images, true masks, predicted masks
    '''
    model.eval() # make sure model in evaluation part
    images_shown = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs) > 0.5

            for i in range(images.size(0)):
                if images_shown >= num_examples:
                    return

                processed_output = keep_largest_regions(outputs[i].cpu().numpy(), n_largest=2)
                processed_output = smooth_lung_mask(processed_output,structure_size=2)

                # Define the figsize
                plt.figure(figsize=(12, 4))
                # Original Image
                plt.subplot(1, 3, 1)
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.title('Original Image')
                plt.axis('off')

                # True Mask
                plt.subplot(1, 3, 2)
                plt.imshow(masks[i].cpu().squeeze(), cmap='gray')
                plt.title('True Mask')
                plt.axis('off')

                # Predicted Mask after processing
                plt.subplot(1, 3, 3)
                plt.imshow(processed_output, cmap='gray')
                plt.title('Processed Predicted Mask')
                plt.axis('off')

                plt.show()
                images_shown += 1 #count the image shown


def visualization_overleaf(model, data_loader, num_examples=5):
    '''
    #define function to show original images, true masks, predicted masks and overlay of true masks and predicted masks
    '''
    model.eval() # make sure model in evaluation part
    images_shown = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs) > 0.5

            for i in range(images.size(0)):
                if images_shown >= num_examples:
                    return

                processed_output = keep_largest_regions(outputs[i].cpu().numpy(), n_largest=2)
                processed_output = smooth_lung_mask(processed_output, structure_size=2)

                # Calculate the Intersection Ratio
                intersection_ratio = np.sum(masks[i].cpu().numpy() * processed_output) / np.sum(masks[i].cpu().numpy() == 1)

                # define the figsize
                plt.figure(figsize=(16, 4))
                # show Original Image
                plt.subplot(1, 5, 1)
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.title('Original Image')
                plt.axis('off')

                # show True Mask
                plt.subplot(1, 5, 2)
                plt.imshow(masks[i].cpu().squeeze(), cmap='Blues')
                plt.title('True Mask')
                plt.axis('off')

                # show Predicted Mask
                plt.subplot(1, 5, 3)
                plt.imshow(processed_output, cmap='Reds')
                plt.title('Predicted Mask')
                plt.axis('off')

                # Overlay of True and Predicted Mask with Intersection Ratio
                plt.subplot(1, 5, 4)
                plt.imshow(masks[i].cpu().squeeze(), cmap='Blues', alpha=0.5)
                plt.imshow(processed_output, cmap='Reds', alpha=0.5)
                plt.title('Overlay\nIntersection Ratio: {:.4f}'.format(intersection_ratio))
                plt.axis('off')

                plt.show()
                images_shown += 1


# Call the function
visualize_model(model, val_loader)
visualization_overleaf(model, val_loader)
