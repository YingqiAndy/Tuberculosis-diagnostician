import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage import color
from skimage import transform
from torchvision import transforms
from classification.resnet_model import DeepResNet

# Load our model
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classify_model = DeepResNet().to(dvc)
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
# Reminder: You need to change the following path to the full path from your computer to the file.
classify_model.load_state_dict(torch.load('classification/classifier.pth', map_location=torch.device('cpu')))
classify_model.eval()
#/Users/bob/pythonProject/BIA4_Group7/
'''
This py file is the interface of classification model and GUI, 
which apply model to process  the input figure in GUI and output its classification (normal or tb)
'''

def classify_lung_xrays(image_paths, model, dvc):
    """
    Classify the given X-ray images into 2 types (normal or tb)

    image_paths: lung X-ray images paths
    model: The loaded UNet model for classification.
    dvc: The device to run the model on (e.g., 'cuda' or 'cpu').
    return: 0:normal   1: tb
    """
    for image_path in image_paths:
        # Load and preprocess the image
        image = plt.imread(image_path)
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image = color.rgb2gray(image)
        image = transform.resize(image, (224, 224))
        image = color.gray2rgb(image)
        image = image.astype(np.float32)
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transformer(image).unsqueeze(0).to(dvc)
        # Call the model
        with torch.no_grad():
            output = model(image_tensor)
        _, predicted = torch.max(output, 1)

        return predicted.item()
#
#
# if __name__ == "__main__":
#     import os
#
#     # Load our model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     classify_model = DeepResNet().to(device)
#     classify_model.load_state_dict(torch.load('classifier.pth', map_location=torch.device('cpu')))
#     classify_model.eval()
#
#     img_dir = "../Xray_image/Dataset2/Tuberculosis/"
#     img_list = os.listdir(img_dir)
#     img_path = ["../Xray_image/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-8.png"]
#     sums = 0
#     label_list = []
#     for i in img_list:
#         i = [img_dir + i]
#         label = classify_lung_xrays(i, classify_model, device)
#         label_list.append(label)
#         sums += label
#     print(sums, sums / len(img_list), len(label_list), label_list)

#     for i in img_list:
#         i = [img_dir + i]
#         label = classify_lung_xrays(i, classify_model, device)
#         label_list.append(label)
#         sums += label
#     print(sums, sums / len(img_list), len(label_list), label_list)
