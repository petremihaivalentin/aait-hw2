import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import argparse
import os
import time
from tqdm import tqdm
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from utils_torchvision import CustomImageDataset

import copy


class CustomValidationImageDataset(Dataset):
    def __init__(self, img_dir, image_paths, master_model, transform=None, target_transform=None):
        self.model = master_model
        self.img_dir = img_dir
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
          image = read_image(self.img_dir + self.image_paths[idx])
          image = self.transform(image)
          self.model.eval()
          with torch.no_grad():
            output = self.model(image.to(device).unsqueeze(0).float())
            return self.image_paths[idx], torch.argmax(output)
        except:
          # print(f"problem with: {self.img_dir + self.image_paths[idx]}")
          pass


if __name__ == '__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  parser = argparse.ArgumentParser(description='Driver py module for HW2')
  parser.add_argument('-m','--model', help='choose model name from [resnet50, mobilenet_v2]', required=True)
  parser.add_argument('-w','--weights', help='weights name', required=True)
  parser.add_argument('-d','--datadir', help='path to val data dir', required=True)
  args = vars(parser.parse_args())

  img_dir = args['datadir']

  if args['model'] == 'resnet50':
    model = models.resnet50().to(device)
  if args['model'] == 'resnet101':
    model = models.resnet101().to(device)
  elif args['model'] == 'mobilenet_v2':
    model = models.mobilenet_v2().to(device)


  image_paths = []
  for filename in os.listdir(img_dir):
    if '.jpeg' in filename:
      image_paths.append(filename)

  model_state = args['weights']

  model_state_dict = torch.load(f'./weights/{model_state}.pth')
  model.load_state_dict(model_state_dict)

  training_dataset_unlabeled = CustomValidationImageDataset(img_dir, image_paths, model, transform = transforms.Compose([
          transforms.Resize((224,224)),
          transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
          transforms.RandomHorizontalFlip()
      ]))
  train_dataloader_unlabeled = DataLoader(training_dataset_unlabeled, batch_size=1, shuffle=True)

  import random

  result = {
      "sample": [],
      "label": [],
      "idx": []
  }

  wrong = []

  for i in tqdm(range(len(training_dataset_unlabeled))):
    try:
      res = training_dataset_unlabeled[i]
      result['sample'].append(res[0])
      result['idx'].append(res[0][:-5])
      result['label'].append(res[1].detach().cpu().numpy())
    except Exception as e:
      # print(e)
      result['sample'].append(image_paths[i])
      result['idx'].append(image_paths[i][:-5])
      result['label'].append(random.randint(0,99))
      wrong.append(i)

  result_df = pd.DataFrame(result)

  result_df['idx'] = pd.to_numeric(result_df['idx'])

  df = result_df.sort_values('idx')

  final_df = df[['sample', 'label']]

  final_df.to_csv(f"{model_state}.csv", index=False)