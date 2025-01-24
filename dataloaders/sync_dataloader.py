
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

from torchvision.transforms import ToTensor, Resize, Compose

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_select=None):
        self.concept_list = list('0123456789ABDEFHNQRT')
        self.concept_dict = dict()
        for i, c in enumerate(self.concept_list):
            self.concept_dict[c] = i
        print(self.concept_dict)

        self.root_dir = root_dir
        self.transform = transform
        self.superclass_names = sorted(os.listdir(root_dir))
        print(self.superclass_names)
        if class_select == None:
            self.class_names = []
            for sc in self.superclass_names:
                self.class_names += sorted(os.listdir(os.path.join(root_dir, sc)))
        else:
            self.class_names = class_select
        print(self.class_names)

        self.name_class_dict = dict()
        self.class_name_dict = dict()
        for i, c in enumerate(self.class_names):
            self.name_class_dict[c] = i
            self.class_name_dict[i] = c
        print(self.class_name_dict)

        self.data = []  # 存储图像路径和对应的标签
        for root, dir, file in os.walk(root_dir):
            if dir != []:
                # print(root, dir, file)
                for cla in dir:
                    if cla in self.class_names:
                        new_root = os.path.join(root, cla)
                        file_list = os.listdir(new_root)
                        for f in file_list:
                            img_path = os.path.join(new_root, f)
                            concepts = [self.concept_dict[con] for con in list(cla)]
                            self.data.append((img_path, self.name_class_dict[cla], concepts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, concepts = self.data[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        concept_tensor = torch.zeros((20,))
        for c in concepts:
            concept_tensor[c] = 1
        return image, label, concept_tensor