import argparse
import numpy as np
import os
import torch
import torchvision
import gdown

# from pathlib import Path
from pytorch_lightning import seed_everything
from torchvision import transforms

SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

print(len(SELECTED_CONCEPTS), len(CONCEPT_SEMANTICS))

# concept_dict = {i: CONCEPT_SEMANTICS[i] for i in range(len(CONCEPT_SEMANTICS))}

# get classes for each 8 real concepts
def _binarize(concepts, selected, width):
    result = []
    binary_repr = []
    concepts = concepts[selected]
    for i in range(0, concepts.shape[-1], width):
        binary_repr.append(
            str(int(np.sum(concepts[i : i + width]) > 0))
        )
    return int("".join(binary_repr), 2)

def load_data(args):
    root_dir = args.dataroot
    BATCH_SIZE = args.batch_size
    img_size = args.img_size
    NUM_WORKERS = args.workers
    device = args.device
    num_concepts = args.num_concepts
    num_hidden = args.num_hidden
    seed = args.seed
    seed_everything(seed)

    # download celeba dataset
    celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            # target_transform=lambda x: x[0].long() - 1,
            # target_type=['attr'],
        )

    # celeba_train_data.target_transform
    # concept selection
    concept_freq = np.sum(celeba_train_data.attr.cpu().detach().numpy(), axis=0) / celeba_train_data.attr.shape[0]
    # print(f"Concept frequency is: {concept_freq}")
    sorted_concepts = list(map(
        lambda x: x[0],
        sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
    ))
    print(sorted_concepts, '\n', concept_freq[sorted_concepts])
    # num_concepts = 6
    concept_idxs = sorted_concepts[:num_concepts]
    concept_idxs = sorted(concept_idxs)
    concept_names = [CONCEPT_SEMANTICS[i] for i in concept_idxs]
    print('concepts', concept_idxs, concept_names)
    hidden_concepts = sorted(
                sorted_concepts[
                    num_concepts:min(
                        (num_concepts + num_hidden),
                        len(sorted_concepts)
                    )
                ]
            )
    print('hidden concepts', hidden_concepts, [CONCEPT_SEMANTICS[i] for i in hidden_concepts])

    # transform
    celeba_train_data.transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_train_data.target_transform = lambda x: [
        torch.tensor(
            _binarize(
                x[1].cpu().detach().numpy(),
                selected=(concept_idxs + hidden_concepts),
                width=1,
            ),
            dtype=torch.long,
        ),
        x[1][concept_idxs].float(),
    ]

    celeba_train_data.target_type = ['identity', 'attr']
    print('data[0]', celeba_train_data[0])

    label_remap = {}
    vals, counts = np.unique(
        list(map(
            lambda x: _binarize(
                x.cpu().detach().numpy(),
                selected=(concept_idxs + hidden_concepts),
                width=1,
            ),
            celeba_train_data.attr
        )),
        return_counts=True,
    )

    for i, label in enumerate(vals):
        label_remap[label] = i

    celeba_train_data.target_transform = lambda x: [
        torch.tensor(
            label_remap[_binarize(
                x[1].cpu().detach().numpy(),
                selected=(concept_idxs + hidden_concepts),
                width=1,
            )],
            dtype=torch.long,
        ),
        x[1][concept_idxs].float(),
    ]
    num_classes = len(label_remap)
    print(num_classes)
    print(celeba_train_data.attr.shape, celeba_train_data.identity.shape)
    print(celeba_train_data[0]) 
    # print(vals, counts)
    factor = args.subsample
    if factor != 1:
        train_idxs = np.random.choice(
            np.arange(0, len(celeba_train_data)),
            replace=False,
            size=len(celeba_train_data)//factor,
        )
        # logging.debug(f"Subsampling to {len(train_idxs)} elements.")
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    print(celeba_train_data.__len__())

    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    print(
        f"Data split is: {total_samples} = {train_samples} (train) + "
        f"{test_samples} (test) + {val_samples} (validation)"
    )
    celeba_train_data, celeba_test_data, celeba_val_data = \
        torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )
    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    dataloaders = {
        'train': train_dl,
        'test': test_dl,
        'val': val_dl,
    }

    return dataloaders, concept_names# , hidden_concepts
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='celeba')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=64)
    # parser.add_argument("--pklroot", type=str, default='celeba')
    parser.add_argument('--workers', type=int, default=8)
    # parser.add_argument('-color-jittered', type=bool, default=True)
    # parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=False)
    # parser.add_argument('--normalized', type=bool, default=False)
    # parser.add_argument('--used-group', type=list, default=None)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num-concepts', type=int, default=6)
    parser.add_argument('--num-hidden', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample', type=int, default=12)
    args = parser.parse_args()
    dataloaders, concept_names = load_data(args)
    print(dataloaders, concept_names)
    train_iterator = iter(dataloaders['train'])
    batch = next(train_iterator)
    inputs, targets = batch[0], batch[1]
    print(inputs.shape, targets.shape)
