from torch.utils.data import Dataset
from PIL import Image


class LearningDataset(Dataset):
    """
    Dataset class, using for loading training & testing data
    """
    def __init__(self, imgpaths, labels, transform=None):
        super(Dataset, self).__init__()
        self.imgpaths = imgpaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, idx):
        img_path = self.imgpaths[idx]
        image = Image.open(img_path).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
