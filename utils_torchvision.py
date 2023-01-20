import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

def _hw1_get_labels_as_pd():
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    train_labels_dataset = this_file_path + "/../task1/task1/train_data/annotations.csv"

    train_labels = pd.read_csv(train_labels_dataset)

    return train_labels


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        print("Reading images...")
        for index, row in tqdm(self.img_labels.iterrows()):
          img_path = os.path.join(self.img_dir, row[0])
          image = read_image(img_path).float()
          self.img_labels.loc[index, 2] = image.shape[0]
        self.img_labels = self.img_labels.loc[self.img_labels[2] == 3]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
          img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
          image = read_image(img_path).float()
          label = self.img_labels.iloc[idx, 1]
          if self.transform:
              image = self.transform(image)
          if self.target_transform:
              label = self.target_transform(label)
          return image, label
        except Exception as e:
          print(e)