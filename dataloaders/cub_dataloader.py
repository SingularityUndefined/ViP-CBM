import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import os
import pickle
from PIL import Image
from torchvision import transforms
import argparse

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import numpy as np
import matplotlib.pyplot as plt
import random

class CUB_dataset(Dataset):
    def __init__(self, pkl_file, data_root, image_transforms, used_group=None) -> None:
        super().__init__()
        with  open(pkl_file, 'rb') as f:
            self.pkl = pickle.load(f)
        cub_root = os.path.join(data_root, 'CUB_200_2011')
        self.img_dir = os.path.join(cub_root, 'images')
        self.img_transforms = image_transforms
        self.n_concepts = 112
        self.n_labels = 200
        self.used_group = used_group

        self.index2class = {}
        self.class2index = {}
        with open(os.path.join(cub_root, 'classes.txt'), 'r') as f:
            for line in f:
                c = line.split(' ')[1].replace('\n', '')
                idx = int(line.split(' ')[0])
                # zero-indexed
                self.index2class[idx-1] = c
                self.class2index[c] = idx - 1
        # attr dict
        attr2attrlabel = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91,
                      93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181,
                      183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
                      254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
        attr_group_dict = {}
        attr_group_dict_name = {}
        self.attr2index, self.index2attr = {}, {}
        with open(os.path.join(data_root, 'attributes.txt'), 'r') as f:
            strings = f.readlines()
        # attr dict
        for i, idx in enumerate(attr2attrlabel):
            label = strings[idx].split(' ')[-1].replace('\n', '')
            group = label.split('::')[0]
            label1 = label.split('::')[1]
            self.attr2index[label] = i
            self.index2attr[i] = label

            if group in attr_group_dict.keys():
                attr_group_dict[group].append(i)
                attr_group_dict_name[group].append(label1)
            else:
                attr_group_dict[group] = [i]
                attr_group_dict_name[group] = [label1]

        self.attr_group_dict = attr_group_dict
        self.attr_group_dict_name = attr_group_dict_name
        self.group_size = [len(val) for val in self.attr_group_dict.values()]
            # print('group_dict', attr_group_dict)
        # used group
        if self.used_group is not None:
            self.idxs = []
            self.group_size = []
            for key in self.attr_group_dict.keys():
                if key in used_group:
                    self.idxs += self.attr_group_dict[key]
                    self.group_size.append(len(self.attr_group_dict[key]))

    def __len__(self):
        return len(self.pkl)
    
    def __getitem__(self, index):
        name = self.pkl[index]['img_path']
        # trim filename
        if 'images' in name:
            name = name.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/images/' ,'')
        img_path = os.path.join(self.img_dir, name)
        image = Image.open(img_path).convert('RGB')
        if self.img_transforms is not None:
            image = self.img_transforms(image)

        concept = torch.Tensor(self.pkl[index]['attribute_label'])
        class_label = torch.Tensor([self.pkl[index]['class_label']])
        concept_certainty = torch.Tensor(self.pkl[index]['attribute_certainty'])

        # group_selection
        if self.used_group:
            concept = concept[self.idxs]
            concept_certainty = concept_certainty[self.idxs]

        sample = {}
        sample['image'] = image
        sample['concept_label'] = concept
        sample['class_label'] = class_label
        sample['concept_certainty'] = concept_certainty
        sample['imageID'] = img_path

        return sample
    
    def get_concept_imbalance_ratio(self):
        num_attr = torch.zeros(112)
        # cocnept counts
        for data in self.pkl:
            num_attr += torch.Tensor(data['attribute_label'])
        
        imbalance_ratio = len(self.pkl) / num_attr - 1 # +1e-6
        return imbalance_ratio
    
    def get_class_imbalance_ratio(self):
        num_class = torch.zeros(200)
        # cocnept counts
        for data in self.pkl:
            num_class[data['class_label']] += 1
        
        imbalance_ratio = len(self.pkl) / num_class - 1 # +1e-6
        return imbalance_ratio
    
# imbalanced sampler, set each batch to be class-equaled
# not used yet
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['class_label']#[0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples
    

def load_data(args):
    data_root = args.dataroot
    pkl_root = args.pklroot
    batch_size = args.batch_size
    resol = args.img_size
    workers = args.workers
    USE_IMAGENET_INCEPTION = args.USE_IMAGENET_INCEPTION
    normalized = args.normalized
    used_group = args.used_group
    color_jittered = args.color_jittered
    # 112 used attrs by CUB, 0-indexed (https://github.com/yewsiang/ConceptBottleneck/issues/15)
    
    train_dataset, val_dataset, test_dataset = None, None, None
    drop_last = False # True?
    resized_resol = int(resol * 256 / 224) # 256?
    if USE_IMAGENET_INCEPTION:
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        mean = [0.5, 0.5, 0.5]
        std = [2, 2, 2]

    trainTransform = transforms.Compose([ # colors with shadows, in PCBM is 0.5
        transforms.RandomHorizontalFlip(),
        transforms.Resize((resized_resol, resized_resol)),
        transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)), # why? or centercrop
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
        ])

    testTransform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
        ])
    
    if color_jittered:
        trainTransform = transforms.Compose([transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),trainTransform])
    
    if normalized:
        trainTransform = transforms.Compose([trainTransform, transforms.Normalize(mean=mean, std=std)])
        testTransform = transforms.Compose([testTransform, transforms.Normalize(mean=mean, std=std)])

    print('start constructing dataset')
    
    # cub_root = os.path.join(data_root, 'CUB_200_2011')
    # image_dir = os.path.join(cub_root, 'images')
    train_pkl = os.path.join(pkl_root, 'train.pkl')
    val_pkl = os.path.join(pkl_root, 'val.pkl')
    test_pkl = os.path.join(pkl_root, 'test.pkl')

    train_dataset = CUB_dataset(train_pkl, data_root, trainTransform, used_group=used_group)
    val_dataset = CUB_dataset(val_pkl, data_root, testTransform, used_group=used_group)
    test_dataset = CUB_dataset(test_pkl, data_root, testTransform, used_group=used_group)

    i = random.randint(0, 100)
    sample = train_dataset.__getitem__(i)
    print('train', i, sample['class_label'], sample['concept_label'], sample['imageID'])
    train_label = train_dataset.index2class[int(sample['class_label'][0])]
    train_image = sample['image'].permute(1,2,0).numpy()
    fig = plt.figure()
    plt.imshow(Image.fromarray((train_image *255).astype(np.uint8)))
    if not os.path.exists('./dataloaders/sample_images'):
        os.makedirs('./dataloaders/sample_images')
    plt.savefig('./dataloaders/sample_images/train_sample_'+train_label+'.png')

    # print(train_dataset.__getitem__(0))
    train_loader, val_loader, test_loader = None, None, None
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, train_dataset.attr2index, train_dataset.class2index, train_dataset.attr_group_dict, train_dataset.group_size

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataroot", type=str, default='CUB')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pklroot", type=str, default='CUB/class_attr_data_10')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=True)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--used-group', type=list, default=None)
    args = parser.parse_args()
    args.used_group = ['has_bill_shape', 'has_wing_color']
    dataloaders, attr2index, class2index, attr_group_dict, group_size = load_data(args)
    print(attr2index)
    print(class2index)
    print(attr_group_dict)
    print(group_size)
    train_sample = next(iter(dataloaders['train']))
    print((train_sample))
'''


    



