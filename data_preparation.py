import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import skimage
from skimage import exposure, filters, color
from sklearn.model_selection import train_test_split

# Set the image size to 512x512
IMAGE_SIZE = (512, 512)

# load image path
#normal_images_path = 'path_to_normal_images'
#tb_images_path = 'path_to_tb_images'
normal_images_path = 'data/TB_Chest_Radiography_Database/Normal'  # Normal images
tb_images_path = 'data/TB_Chest_Radiography_Database/Tuberculosis'  # Tuberculosis images

#set augmented images path
tb_augmented_dir = 'data/augmented_images/tb'
#normal_augmented_dir = 'augmented_images/normal'  # augmentation data save path for normal image

#set processed images path
processed_save_path_normal = 'data/processed_images/normal'  # The path to save the processed normal image
processed_save_path_tb = 'data/processed_images/tb'  # The path to save the processed tb image

def load_and_preprocess_images(images_path, image_size, save_path):
    """Load and preprocess the image, and save the processed image to disk"""
    images = []
    for img_index, img_name in enumerate(os.listdir(images_path)):
        img_path = os.path.join(images_path, img_name)
        img = plt.imread(img_path)
        if img.ndim > 2:
            if img.shape[-1] == 3 and img.ndim == 3:
                img = color.rgb2gray(img)  # If it is an RGB image, convert to grayscale image
            else:
                return -1
        img = skimage.transform.resize(img, image_size)  # Resize the image to a preset parameter
        #Process multi-channel images to convert them into grayscale images.

        # Enhance image contrast with histogram
        img = exposure.equalize_hist(img)
        # Gaussian blur is used to reduce image noise to obtain smoother, more distinctive pictures.
        img = filters.gaussian(img, sigma=1)
        # Normalize the image and scale each pixel value to the [0, 1] interval
        img = img / np.max(img)

        save_image(img, save_path, f"{img_name}_processed.png")
        images.append(img)
    return np.array(images)

def load_images_from_folder(folder,image_size):
    """Load the image from the folder"""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):#Specifies the image format to be loaded.
            img_path = os.path.join(folder, filename)

            img = plt.imread(img_path)
            if img.ndim > 2:
                if img.shape[-1] == 3 and img.ndim == 3:
                    img = color.rgb2gray(img)  # Convert multi-channel images into grayscale images
                else:
                    return -1
            img = skimage.transform.resize(img, image_size)  # Resize the image to the preset parameter

            images.append(img)
    return np.array(images)

def data_augmentation(image_array, img_index, output_dir, image_type, enhance=True):
    """Perform data augmentation and save the images"""
    img = Image.fromarray((image_array * 255).astype('uint8'))
    augmented_images = []

    if enhance:
        # Flip the image horizontally and save the image for augmenting the data.
        flipped_img = ImageOps.mirror(img)
        flipped_img_array = np.array(flipped_img) / 255
        flipped_img_name = f'{image_type}_flipped_{img_index+1}.png'
        save_image(flipped_img_array, output_dir, flipped_img_name)
        augmented_images.append(flipped_img_array)

        # Randomly adjust the brightness and save the image for augmenting the data.
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(np.random.uniform(0.7, 1.3))
        bright_img_array = np.array(bright_img) / 255
        bright_img_name = f'{image_type}_brightness_{img_index+1}.png'
        save_image(bright_img_array, output_dir, bright_img_name)
        augmented_images.append(bright_img_array)

        # Random contrast adjustment and save the image for augmenting the data.
        enhancer = ImageEnhance.Contrast(bright_img)
        contrast_img = enhancer.enhance(np.random.uniform(0.7, 1.3))
        contrast_img_array = np.array(contrast_img) / 255
        contrast_img_name = f'{image_type}_contrast_{img_index+1}.png'
        save_image(contrast_img_array, output_dir, contrast_img_name)
        augmented_images.append(contrast_img_array)

        # Random rotation and save the image for augmenting data
        rotation_degree = np.random.uniform(-6, 6)  # Rotate the Angle randomly between -15 and 15 degrees
        rotated_img = img.rotate(rotation_degree)
        rotated_img_array = np.array(rotated_img) / 255
        rotated_img_name = f'{image_type}_rotated_{img_index+1}.png'
        save_image(rotated_img_array, output_dir, rotated_img_name)
        augmented_images.append(rotated_img_array)

    return np.array(augmented_images)

def save_image(img_array, output_dir, img_name):
    """Save the image to specific directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_path = os.path.join(output_dir, img_name)
    img = Image.fromarray((img_array * 255).astype('uint8'))
    img.save(img_path)#Save the image to the output directory

# implement image process
normal_images_processed = load_and_preprocess_images(normal_images_path, IMAGE_SIZE, processed_save_path_normal)
tb_images_processed = load_and_preprocess_images(tb_images_path, IMAGE_SIZE, processed_save_path_tb)


# Apply data augmentation and save
# normal image
# for i, img in enumerate(normal_images_processed):
#     data_augmentation(img, i, normal_augmented_dir, 'Normal')

# tb image
for i, img in enumerate(tb_images_processed):
    data_augmentation(img, i, tb_augmented_dir, 'Tuberculosis')

'''
In order to ensure that the data after preprocessing is saved correctly, we made a confirmation. And visualize the image to determine success.
'''
# #Load the augmented image
# #normal_images_aug = load_images_from_folder(normal_augmented_dir,IMAGE_SIZE)
# tb_images_aug = load_images_from_folder(tb_augmented_dir,IMAGE_SIZE)
#
# # Creating tags
# normal_labels = np.zeros(len(normal_images_processed))
# tb_labels = np.ones(len(tb_images_processed) + len(tb_images_aug))
#
# # Merge original and augmented images
# X = np.concatenate((normal_images_processed, tb_images_processed, tb_images_aug), axis=0)
# y = np.concatenate((normal_labels, tb_labels), axis=0)
#

# print(f'Whole data set shape: {X.shape}')

# #visualization
# #processed_images = load_and_preprocess_images(normal_images_path, IMAGE_SIZE)  # OR: tb_images_path
# processed_images = normal_images_aug  # OR: tb_images_path
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(processed_images[i], cmap='gray')
#     plt.title(f"Processed image {i+1}")
#     plt.axis('off')
# plt.show()