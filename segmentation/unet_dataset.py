import torch
from torch.utils.data import Dataset
from skimage import color
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import transform
class XRayDataset(Dataset):
    """
    Load the lung X-ray images with their corresponding masks and set this as a group
    """
    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # load files in a sorted manner

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        """
        Gets the images and masks in the dataset according to the index.

        :return: processed images and masks
        """
        img_file = self.images[idx]
        # use 'split' to get the specific part in file name that we want
        img_number = img_file.split('_')[-1].split('.')[0]

        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, f'cxrmask_{img_number}.jpeg')

        image = plt.imread(img_path)
        mask = plt.imread(mask_path)
        new_size = (256, 256)  #resize
        image = transform.resize(image, new_size)
        mask = transform.resize(mask, new_size)


        # Check the number of image channels, if 4, convert to 3
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # if RGB, convert to gray
        if image.ndim == 3:
            image = color.rgb2gray(image)
        if mask.ndim == 3:
            mask = color.rgb2gray(mask)
        # convert mask to binary images
        mask = (mask > 0.5).astype(float)

        # apply get_transform()
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Convert images to PyTorch Tensor format and do normalization
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
