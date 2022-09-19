import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

class DigitsDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform):
    self.csv_file = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.image_names = self.csv_file[:]['Imagem']
    self.labels = np.array(self.csv_file.drop(['Imagem', 'Valor'], axis = 1))
  
  def __len__(self):
    return len(self.csv_file)
  
  def __getitem__(self, index):
    image = cv2.imread(self.root_dir + self.image_names.iloc[index])
    image = self.transform(image)
    targets = self.csv_file.values[index][1]
    return image, targets
