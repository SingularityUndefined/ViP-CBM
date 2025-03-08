import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import argparse
import random
import matplotlib.pyplot as plt

def get_class_index_dict(data_root, set_zero=True):
    filepath = os.path.join(data_root, 'classes.txt')
    class2index = dict()
    index2class = dict()
    with open(filepath, 'r') as f:
        for line in f:
            class_index = int(line.split('\t')[0].strip())
            class_name = line.split('\t')[1].strip()
            # starts from zero
            class2index[class_name] = class_index - int(set_zero)
            index2class[class_index-int(set_zero)] = class_name
    f.close()
    return class2index, index2class

def get_concept_index_dict(data_root, set_zero=True):
    filepath = os.path.join(data_root, 'predicates.txt')
    class2index = dict()
    index2class = dict()
    with open(filepath, 'r') as f:
        for line in f:
            class_index = int(line.split('\t')[0].strip())
            class_name = line.split('\t')[1].strip()
            # starts from zero
            class2index[class_name] = class_index - int(set_zero)
            index2class[class_index-int(set_zero)] = class_name
    f.close()
    return class2index, index2class

data_root = 'AwA2/Animals_with_Attributes2'
# class2index, index2class = get_class_index_dict('AwA2/Animals_with_Attributes2')
# print(class2index, index2class)

# c to y
def get_label_concept_binary(data_root):
    filepath = os.path.join(data_root, 'predicate-matrix-binary.txt')
    arr = np.genfromtxt(filepath)
    return arr

def get_label_concept_certainty(data_root):
    filepath = os.path.join(data_root, 'predicate-matrix-continuous.txt')
    arr = np.genfromtxt(filepath)
    return arr

# constant numbers

class AwA2_Dataset(Dataset):
    def __init__(self, data_root, used_group=None, train_transforms=None, test_transforms=None, is_train=True, set_zero=True):
        super().__init__()
        self.set_zero = set_zero
        self.is_train = is_train
        self.data_root = data_root
        self.used_group = used_group

        self.class2index, self.index2class = get_class_index_dict(data_root, set_zero)
        self.concept2index, self.index2concept = get_concept_index_dict(data_root, set_zero)
        self.label_concept_binary = get_label_concept_binary(data_root)
        self.label_concept_certainty = get_label_concept_certainty(data_root)

        group_count = [8, 3, 2, 1, 2, 2, 5, 1, 1, 1, 4, 1, 1, 1, 1, 5,2,2,1, 2,2, 3,5,6,2, 14, 2,1,2,2]
        group_name = ['color', 'pattern', 'hair', 'skin', 'size', 'fat', 'hand', 'leg', 'neck', 'tail', 'teeth', 'horn', 'claw', 'tusk', 'smelly', 'moves', 'speed', 'strength', 'muscles', 'walk', 'active', 'habbits', 'food', 'role', 'world', 'live','nature', 'smart', 'family', 'spot']
        concept_list = list(self.concept2index.keys())
        group_idx_dict = {}
        group_name_dict = {}
        checknum = 0
        for i in range(len(group_name)):
            group_idx_dict[group_name[i]] = list(range(checknum, checknum + group_count[i]))
            group_name_dict[group_name[i]] = concept_list[checknum:checknum + group_count[i]]
            checknum += group_count[i]
        self.group_idx_dict = group_idx_dict
        self.group_name_dict = group_name_dict

        self.group_size = group_count
        if self.used_group is not None:
            idxs = []
            self.group_size = []
            for key in self.group_idx_dict.keys():
                if key in self.used_group:
                    idxs += self.group_idx_dict[key]
                    self.group_size.append(len(self.group_idx_dict[key]))
            self.label_concept_binary = self.label_concept_binary[:,idxs]
            self.label_concept_certainty = self.label_concept_certainty[:,idxs]

        self.image_dir = os.path.join(data_root, 'JPEGImages')
        self.image_list_file = os.path.join(data_root, 'Features/ResNet101/AwA2-filenames.txt')
        self.label_list_file = os.path.join(data_root, 'Features/ResNet101/AwA2-labels.txt')
        self.feature_list_file = os.path.join(data_root, 'Features/ResNet101/AwA2-features.txt')

        self.features = np.genfromtxt(self.feature_list_file)
        # set zero
        self.labels = np.genfromtxt(self.label_list_file, dtype=int) - int(set_zero)
        self.images = pd.read_csv(self.image_list_file, header=None)
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        filename = self.images.iloc[index, 0]
        # print(filename, label)
        folder_name = filename.split('_')[0].strip()
        file_path = os.path.join(self.image_dir, folder_name, filename)
        image = Image.open(file_path).convert('RGB')
        if self.is_train:
            if self.train_transforms is not None:
                image = self.train_transforms(image)
        else:
            if self.test_transforms is not None:
                image = self.test_transforms(image)
        if self.set_zero:
            concept = torch.Tensor(self.label_concept_binary[label])
            certainty = torch.Tensor(self.label_concept_certainty[label])
        else:
            concept = torch.Tensor(self.label_concept_binary[label-1])
            certainty = torch.Tensor(self.label_concept_certainty[label-1])
        
        # print concept before selections

        # print(label)
        sample = {}
        sample['image'] = image
        sample['imageID'] = file_path
        sample['feature'] = feature
        sample['concept_label'] = concept
        sample['concept_certainty'] = certainty
        sample['class_label'] = torch.Tensor([label])

        return sample

# load data, with random split into train val test
# define transforms

def load_data(args):
    data_root = args.dataroot
    batch_size = args.batch_size
    resol = args.img_size
    workers = args.workers
    train_val_test_ratio = args.train_val_test_ratio
    USE_IMAGENET_INCEPTION = args.USE_IMAGENET_INCEPTION
    used_group = args.used_group
    normalized = args.normalized
    # 112 used attrs by CUB, 0-indexed (https://github.com/yewsiang/ConceptBottleneck/issues/15)
    
    drop_last = False # True?
    resized_resol = int(resol * 256 / 224) # 256?
    if USE_IMAGENET_INCEPTION:
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        mean = [0.5, 0.5, 0.5]
        std = [2, 2, 2]

    trainTransform = transforms.Compose([
        transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)), # colors with shadows, in PCBM is 0.5
        transforms.RandomHorizontalFlip(),
        transforms.Resize((resized_resol, resized_resol)),
        transforms.RandomResizedCrop(resol),# , scale=(0.8, 1.0)), # why? or centercrop
        transforms.ToTensor(),
        ])

    testTransform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
        ])
    if normalized:
        trainTransform = transforms.Compose([trainTransform, transforms.Normalize(mean=mean, std=std)])
        testTransform = transforms.Compose([testTransform, transforms.Normalize(mean=mean, std=std)])

    print('start constructing dataset')
    
    dataset = AwA2_Dataset(data_root, train_transforms=trainTransform, test_transforms=testTransform, is_train=True, used_group=used_group)
    # separate into train, val, test
    train_len, val_len = int(train_val_test_ratio[0] * len(dataset)), int(train_val_test_ratio[1] * len(dataset))
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[train_len, val_len, len(dataset) - train_len - val_len], generator=torch.Generator().manual_seed(3407))
    val_dataset.is_train = False
    test_dataset.is_train = False

    i = random.randint(0, 100)
    train_sample, val_sample = train_dataset.__getitem__(i), val_dataset.__getitem__(i)
    print('train', i, train_sample['class_label'], train_sample['concept_label'], train_sample['imageID'])
    print('val', i, val_sample['class_label'], val_sample['concept_label'], val_sample['imageID'])
    train_label, val_label = dataset.index2class[int(train_sample['class_label'][0])], dataset.index2class[int(val_sample['class_label'][0])]
    train_image, val_image = train_sample['image'].permute(1,2,0).numpy(), val_sample['image'].permute(1,2,0).numpy()
    fig = plt.figure()
    plt.imshow(Image.fromarray((train_image *255).astype(np.uint8)))
    plt.savefig('./dataloaders/sample_images/train_sample_'+train_label+'.png')

    fig = plt.figure()
    plt.imshow(Image.fromarray((val_image *255).astype(np.uint8)))
    plt.savefig('./dataloaders/sample_images/val_sample_'+val_label+'.png')

    # print(train_dataset.__getitem__(0))
    train_loader, val_loader, test_loader = None, None, None
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, dataset.class2index, dataset.concept2index, dataset.group_idx_dict, dataset.group_size# group index dict, concept dict, etc.



##################
# testing codes
##################
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataroot", type=str, default='AwA2/Animals_with_Attributes2')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=True)
    parser.add_argument('--train-val-test-ratio', type=list, default=[0.6, 0.1, 0.3])
    parser.add_argument('--used-group', type=list, default=None)
    parser.add_argument('--normalized', type=bool, default=False)
    args = parser.parse_args()
    args.used_group = ['color', 'pattern']
    dataloaders, class2index, concept2index, group_idx_dict, group_size = load_data(args)
    print(class2index)
    print(concept2index)
    print(group_idx_dict)
    print(group_size)
    train_sample = next(iter(dataloaders['train']))
    print(train_sample)
'''


# def get_label