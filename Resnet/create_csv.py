import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

import pandas as pd


device = ("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/sungsu21/Project/data/Breast-cancer"
train_df = pd.DataFrame(columns= ["img_path","label"])

# files = [f for f in os.listdir(root_dir) if os.path.isfile(f)]
# print(files)
# train_df["img_path"] = os.listdir("/home/sungsu21/Project/data/Breast-cancer")

FILES = []
idx = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        FILES.append(subdir + "/" + file)

train_df["img_path"] = FILES

for idx,file_name in enumerate(FILES):
    print(idx)
    if "class1" in file_name:
        train_df["label"][idx] = 1
    elif "class0" in file_name:
        train_df["label"][idx] = 0

train_df.to_csv(r'train_csv.csv', index = False, header=True)
