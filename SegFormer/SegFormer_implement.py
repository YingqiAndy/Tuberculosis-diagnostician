import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure

# Load our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Prioritize using GPU to run code
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1).to(device) # Use the learned SegFormer model to process images

model.load_state_dict(torch.load('segformer_model.pth'))
model.eval()

# Set working path
normal_image_dir = 'bob/Downloads/PKU_summerschool/data/processed_images/normal'
normal_output_dir = 'bob/Downloads/PKU_summerschool/data/segmented_images/normal_masks'

tb_image_dir = 'bob/Downloads/PKU_summerschool/data/processed_images/tb'
tb_output_dir = 'bob/Downloads/PKU_summerschool/data/segmented_images/tb_masks'

def keep_largest_regions(mask, n_largest=2):
    '''
    The function defined keeps only the two largest connected white areas in the mask (i.e. the two lung areas)
    '''
    if mask.ndim > 2:
        mask = mask.squeeze()
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    unique, counts = np.unique(labeled, return_counts=True)

    largest_components = unique[np.argsort(-counts)][:n_largest + 1]

    filtered_mask = np.isin(labeled, largest_components[1:])
    return filtered_mask.astype(np.float)

def smooth_lung_mask(mask, structure_size=8):
    '''
    Smooth the lung mask, using binary opening and closing to remove small holes or small irregular edges in the mask,
    to make the mask more continuous and consistent
    '''
    binary_structure = generate_binary_structure(2, structure_size)

    mask_opened = binary_opening(mask, structure=binary_structure)
    mask_smoothed = binary_closing(mask_opened, structure=binary_structure)

    return mask_smoothed.astype(np.float64)

def segment_and_save_image(image_path, output_dir, model, device, transform, show_image=False):
    """
    Segment lung X-ray image and save the segmented mask.
    """
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)  # Transform to tensor and add batch dimension

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        output_prob = torch.sigmoid(logits)
        output_binary = (output_prob > 0.5).float()
        output_np = output_binary.cpu().numpy().squeeze()

        segmented_mask = keep_largest_regions(output_np, n_largest=2)
        segmented_mask = smooth_lung_mask(segmented_mask, structure_size=2)

    image_prefix = os.path.splitext(os.path.basename(image_path))[0]

    if show_image:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title(f'Original Image ({image_prefix})')
        ax[0].axis('off')

        ax[1].imshow(segmented_mask, cmap='gray')
        ax[1].set_title(f'Segmented Mask ({image_prefix})')
        ax[1].axis('off')

        plt.show()

    save_image_torch(segmented_mask, output_dir, f'Segmented_{os.path.basename(image_path)}')
    if show_image:
        plt.close(fig)

def save_image_torch(segmented_mask, output_dir, output_file_name):
    """
    Save the segmented mask to the specific directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_file_name)
    plt.imsave(output_path, segmented_mask, cmap='gray')

# Convert the image to a PyTorch Tensor and normalize the operation
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Segment and save images in normal directory
segmented_normal_image_paths = []  # Store the processed image path
for image_file in os.listdir(normal_image_dir):
    image_path = os.path.join(normal_image_dir, image_file)
    segment_and_save_image(image_path, normal_output_dir, model, device, transform, show_image=False)
    segmented_normal_image_paths.append(image_path)

# Display a subset of processed images
max_images_to_show = 50  # Select 50 pictures for display
for i, image_path in enumerate(segmented_normal_image_paths):
    if i < max_images_to_show:
        segment_and_save_image(image_path, normal_output_dir, model, device, transform, show_image=True)
    else:
        break

# Segment and save images in tb directory
segmented_tb_image_paths = []  # Store the processed image path
for image_file in os.listdir(tb_image_dir):
    image_path = os.path.join(tb_image_dir, image_file)
    segment_and_save_image(image_path, tb_output_dir, model, device, transform, show_image=False)
    segmented_tb_image_paths.append(image_path)

# Display a subset of processed images
for i, image_path in enumerate(segmented_tb_image_paths):
    if i < max_images_to_show:
        segment_and_save_image(image_path, tb_output_dir, model, device, transform, show_image=True)
    else:
        break