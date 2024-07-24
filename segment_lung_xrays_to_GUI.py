import os
import numpy as np
import torch
from PIL import Image,ImageOps, ImageFilter
from torchvision import transforms
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure
from segmentation.unet_model import UNet

# Load our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segment_model = UNet(n_channels=1, n_classes=1).to(device)
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
#Reminder: You need to change the following file path to the true path in your computer.
segment_model.load_state_dict(torch.load('segmentation/unet_model.pth', map_location=torch.device('cpu')))
segment_model.eval()   # set the model into evaluation mode
#/Users/bob/pythonProject/BIA4_Group7/


def get_transform():
    """
    Convert the image to a PyTorch Tensor format and do normalization
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
transformer = get_transform()


def keep_largest_regions(mask, n_largest=2):
    '''
    Defined function to only retain the two largest connected white areas in the mask (i.e., the two lung areas)
    '''
    # Ensure the mask is 2D format
    if mask.ndim > 2:
        mask = mask.squeeze()
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, ncomponents = label(mask, structure)
    unique, counts = np.unique(labeled, return_counts=True)

    # Find the indices (location) of the largest components
    largest_components = unique[np.argsort(-counts)][:n_largest + 1]

    # only remain the top 2 largest components and remove others
    filtered_mask = np.isin(labeled, largest_components[1:])
    return filtered_mask.astype(np.float64)


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



def segment_lung_xrays(image_paths, segment_model, device):
    """
    Segment lung regions (get lung masks) from the given X-ray images.
    parameters:
    image_paths: lung X-ray images paths
    model: The loaded UNet model for segmentation.
    device: The device to run the model on (e.g., 'cuda' or 'cpu').
    return: A list of segmented masks
    """
    segmented_masks = []

    for image_path in image_paths:
        # Load and preprocess the image
        image=Image.open(image_path)

        # if is RGBA, convert to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # if is RGB, convert to gray
        if image.mode == 'RGB':
            #histogram equalization
            image = ImageOps.equalize(image)
            #Gaussian Blur
            image = image.filter(ImageFilter.GaussianBlur(1))
            image = image.convert('L')


        img = image.convert('L')
        img = img.resize((512, 512))
        # convert to pytorch tensor
        image_tensor = transformer(img).unsqueeze(0).to(device)
        # Segment the image
        with torch.no_grad():
            output = segment_model(image_tensor)
            output = torch.sigmoid(output) > 0.5
            output_np = output.cpu().numpy().squeeze()
            #  post-processing
            segmented_mask = keep_largest_regions(output_np)
            segmented_mask = smooth_lung_mask(segmented_mask)

        segmented_masks.append(segmented_mask)

    return segmented_masks





