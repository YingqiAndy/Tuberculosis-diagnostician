import numpy as np
import torch
import numpy as np
from torch.utils.data import DataLoader
from unet_model import UNet
from unet_dataset import XRayDataset, get_transform

# Load our model
# 'map_location=torch.device('cpu')': This is to ensure that "CPU" can be used even if your computer does not support "CUDA".
# Reminder: You need to change the following path to the full path from your computer to the file.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('unet_model.pth'))
model.to(device)

#parameter setting
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 40

# load data
#Reminder: You need to change the following file path to the true path in your computer.
transform = get_transform()
test_dataset = XRayDataset(image_dir='../data/200_test_external/image',
                           mask_dir='../data/200_test_external/mask',
                           transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def calculate_iou(preds, true_masks):
    # Convert the prediction and true mask into binary image data (only 0 or 1)
    preds_bin = (preds > 0.5).float()
    masks_bin = (true_masks > 0.5).float()

    # intersection
    intersection = np.sum(masks_bin.cpu().numpy() * preds_bin.cpu().numpy())
    # union
    union = np.sum(masks_bin.cpu().numpy() == 1)
    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)  # 1e-6 is to avoid zeros in the denominator

    return iou.mean()

# Validation Loop
iou_list = []
model.eval()
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        output = model(images)
        output_prob = torch.sigmoid(output)
        iou_score = calculate_iou(output_prob, masks)
        iou_list.append(iou_score.item())

# calculate the average IoU of all batches
average_iou = sum(iou_list) / len(iou_list)
print(f"Average IoU Score: {average_iou}")


#Output:
#Average IoU Score: 0.9548176625887942