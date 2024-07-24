import os
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from resnet_model import DeepResNet
from resnet_train import Training
from classification_Dataset import LearningDataset

EPOCH = 20
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
LR = 1e-6
# Reminder: You need to change the following file path to the true path in your computer.
root_dir = "../data/classification_data"
class_folders = ["normal", "tb"]


def splitting(dataset_dir: str,
              classes: list,
              test_size=0.2,
              tfm=None):
    """

    :param dataset_dir: string. Directory storing the datasets, each of which is one of the classes
    :param classes: list of string. List of classes, where the index of each class in the list is the output of the prediction.
                    E.g. ["normal", "tb"], the class "normal" is predicted as 0, and the class "tb" is predicted as 1.
    :param test_size: float. Percentage of the test dataset
    :param tfm: transforms.Compose. Transforms to be applied to the elements of the dataset
    :return: training datasets and testing datasets. Customised dataset inherited torch.utils.data.Dataset
    """
    filepaths = []
    labels = []

    for label, folder in enumerate(classes):
        full_path: str = os.path.join(dataset_dir, folder)
        img_path = [os.path.join(full_path, f) for f in os.listdir(full_path)]
        filepaths.extend(img_path)
        labels.extend([label] * len(img_path))

    train_filepaths, test_filepaths, train_labels, test_labels = \
        train_test_split(filepaths, labels, test_size=test_size, random_state=42)

    train = LearningDataset(train_filepaths, train_labels, transform=tfm)
    test = LearningDataset(test_filepaths, test_labels, transform=tfm)

    return train, test


def main():
    """
    Main function of the classification
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deep_resnet_model = DeepResNet().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset, test_dataset = splitting(root_dir, class_folders, test_size=0.2, tfm=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    trainer = Training(deep_resnet_model, train_loader, test_loader)
    trainer.set_optimizer(learning_rate=LR)
    trainer.train(num_epochs=EPOCH)
    trainer.evaluate()


if __name__ == '__main__':
    main()
