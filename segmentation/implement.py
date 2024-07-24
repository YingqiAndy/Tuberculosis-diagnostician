import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from torch.utils.data import DataLoader
from unet_dataset import XRayDataset,get_transform
from unet_model import UNet
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure

# Load our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#Prioritize using GPU to run code
model = UNet(n_channels=1, n_classes=1).to(device)#Use the learned U-Net model to process images

model.load_state_dict(torch.load('unet_model.pth'))
# model.load_state_dict(torch.load('unet_model.pth'),map_location=torch.device('cpu'))
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
model.eval()

#Set working path
normal_image_dir = '../data/processed_images/normal'
normal_output_dir = '../data/segmented_images/normal_masks'

tb_image_dir = '../data/processed_images/tb'
tb_output_dir = '../data/segmented_images/tb_masks'



def keep_largest_regions(mask, n_largest=2):
    '''
    The function defined keeps only the two largest connected white areas in the mask (i.e. the two lung areas)
    '''
    # Ensure the mask is 2D format
    if mask.ndim > 2:
        mask = mask.squeeze()
    structure = np.ones((3, 3), dtype=np.int)# Create a structuring element of shape (3, 3) for morphological labeling.
    labeled, ncomponents = label(mask, structure)
    unique, counts = np.unique(labeled, return_counts=True)

    # Find the location of the largest components
    largest_components = unique[np.argsort(-counts)][:n_largest + 1]  # background is usually labeled as 0, so here +1 is necessary

    # only remain the top 2 largest components and remove others
    filtered_mask = np.isin(labeled, largest_components[1:])  # Exclude background with index 0
    return filtered_mask.astype(np.float)


def smooth_lung_mask(mask, structure_size=8):
    '''
    Smooth the lung mask, using binary opening and closing to remove small holes or small irregular edges in the mask,
    to make the mask more continuous and consistent
    '''
    # Create a 2D structural element
    binary_structure = generate_binary_structure(2, structure_size)

    # Apply binary opening and binary closing.
    mask_opened = binary_opening(mask, structure=binary_structure)
    mask_smoothed = binary_closing(mask_opened, structure=binary_structure)
    # This can make the image smoother and make the image features more prominent.

    return mask_smoothed.astype(np.float64)


def segment_and_save_image(image_path, output_dir, model, device, transform, show_image=False):
    """
    Segment lung X-ray image and save the segmented mask.
    """
    # Open the image file
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)  # Transform to tensor and add batch dimension

    #Output results based on the model.
    with torch.no_grad():
        output = model(image_tensor)
        output_prob = torch.sigmoid(output)
        output_binary = (output_prob > 0.5).float()
        output_np = output_binary.cpu().numpy().squeeze()


        segmented_mask = keep_largest_regions(output_np, n_largest=2)
        # Process the mask output from the model to ensure that the two largest connected areas are left.
        segmented_mask = smooth_lung_mask(segmented_mask ,structure_size=2)
        # Smoothing these two areas has resulted in a more accurate mask model.

    # Extract the prefix from the image file name
    image_prefix = os.path.splitext(image_file)[0].split('-')[0]

    # Show the images
    if show_image:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title(f'Original Image ({image_prefix})')  # Use new title to distinguish easily
        ax[0].axis('off')

        ax[1].imshow(segmented_mask, cmap='gray')
        ax[1].set_title(f'Segmented Mask ({image_prefix})')  # Use new title to distinguish easily
        ax[1].axis('off')

        plt.show()

    # Save the processed image
    save_image(segmented_mask, output_dir, f'Segmented_{os.path.basename(image_path)}')
    if show_image:
        plt.close(fig)  # Close the plot to free up memory

def save_image(segmented_mask, output_dir, output_file_name):
    """
    Save the segmented mask to the specific directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it does not exist
    output_path = os.path.join(output_dir, output_file_name)# Form a complete output path.
    plt.imsave(output_path, segmented_mask, cmap='gray')




#Convert the image to a PyTorch Tensor and normalize the operation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#Normal
### Segment and save images in each directory.
segmented_normal_image_paths = []  # Store the processed image path
for image_file in os.listdir(normal_image_dir):
    image_path = os.path.join(normal_image_dir, image_file)
    segment_and_save_image(image_path, normal_output_dir, model, device, transform, show_image=False)
    segmented_normal_image_paths.append(image_path)  # input the processed image path into a list
### Display a subset of processed images
max_images_to_show = 50   #Select 50 pictures for display.
for i, image_path in enumerate(segmented_normal_image_paths):
    if i < max_images_to_show:
        image = Image.open(image_path)
        segment_and_save_image(image_path, normal_output_dir, model, device, transform, show_image=True)
    else:
        break
#Processed and saved each noraml images. The loop stops after the maximum number of displays is reached.
#tb
### segment and save images
segmented_tb_image_paths = []  # Store the processed image path
for image_file in os.listdir(tb_image_dir):
    image_path = os.path.join(tb_image_dir, image_file)
    segment_and_save_image(image_path, tb_output_dir, model, device, transform, show_image=False)
    segmented_tb_image_paths.append(image_path)  # input the processed image path into a list
### Display a subset of processed images
max_images_to_show = 50  #Select 50 pictures for display.
for i, image_path in enumerate(segmented_tb_image_paths):
    if i < max_images_to_show:
        image = Image.open(image_path)
        segment_and_save_image(image_path, tb_output_dir, model, device, transform, show_image=True)
    else:
        break
#Processed and saved each tb images. The loop stops after the maximum number of displays is reached.