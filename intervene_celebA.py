import torch
import torch.nn as nn
from torch import optim
from utils import *
from train.train import train
from models.SECBM import *
from models.baseline_models import *
from dataloaders import celebA_dataloader
from pytorch_lightning import seed_everything

import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
import random
import argparse

from matplotlib import pyplot as plt
# 1. logging settings

def intervene(model, test_dataloader, device, intervene_group):
    '''
    model: baseline model or SemanticCBM
    dataloader: test loader
    '''
    model.eval()

    total_sample = 0
    y_correct = 0
    for samples in tqdm(test_dataloader):
        if not isinstance(samples, dict):
            samples = {
                'image': samples[0],
                'class_label': samples[1][0],
                'concept_label': samples[1][1]
            }
        images, labels, concepts = samples['image'], samples['class_label'], samples['concept_label']
        x, y, c = images.to(device), labels.to(device).squeeze().type(torch.long), concepts.to(device)

        with torch.no_grad():
            y_out = model.intervene(x, c, intervene_group) # a softmax list
            # print(y_out)
            y_pred = torch.argmax(y_out, dim=1)
            y_correct += torch.sum(y_pred == y).item()
            total_sample += y.size(0)

    return y_correct / total_sample
            # compute task accuracy

def test_model(model, test_dataloader, device):
    # print('test original model')
    model.eval()
    total_sample = 0
    y_correct = 0
    for samples in tqdm(test_dataloader):
        if not isinstance(samples, dict):
            samples = {
                'image': samples[0],
                'class_label': samples[1][0],
                'concept_label': samples[1][1]
            }
        images, labels, concepts = samples['image'], samples['class_label'], samples['concept_label']
        x, y, c = images.to(device), labels.to(device).squeeze().type(torch.long), concepts.to(device)

        with torch.no_grad():
            _, y_out = model(x, c) # a softmax list
            # print(y_out)
            y_pred = torch.argmax(y_out, dim=1)
            y_correct += torch.sum(y_pred == y).item()
            total_sample += y.size(0)

    return y_correct / total_sample


def run_intervene(logger, channel, emb_dim, device, model_filename='model-400.pth', shift='none', nonlinear=True, model_name='ViP-CBM', seed=3407, dataset_folder='celeba'):
    n_classes = 256
    n_concepts = 6
    # seed_torch(seed)
    seed_everything(seed)
    log_root = f'./FinalLogs_0224/Grouping'
    # logger_root = 'SE-CBM-group/FinalLogger'
    checkpoint_root = './FinalCheckpoints_0224'

    # changing components
    # dataset_folder = 'CUB'
    NONLINEAR = 'Nonlinear'if nonlinear else 'Linear'
    experiment_folder = f'{channel}_{emb_dim}/{model_name}/Seed_{seed}'
    # logger_name = f'{model_name}_{channel}_{emb_dim}_{NONLINEAR}.log'
    # create logger, log, checkpoints dir
    log_dir = os.path.join(log_root, dataset_folder, experiment_folder)
    checkpoint_dir = os.path.join(checkpoint_root, dataset_folder, experiment_folder)
    # logger_dir = os.path.join(logger_root, dataset_folder)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.makedirs(logger_dir, exist_ok=True)
    # create logger and writer
    writer = SummaryWriter(log_dir)

    # 2. parsers
    # parser = argparse.ArgumentParser(description='manual to this script')
    # parser.add_argument("--dataroot", type=str, default='AwA2/Animals_with_Attributes2')
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--img-size", type=int, default=256)
    # parser.add_argument('--workers', type=int, default=8)
    # parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=True)
    # parser.add_argument('--train-val-test-ratio', type=list, default=[0.6, 0.1, 0.3])
    # parser.add_argument('--used-group', type=list, default=None)
    # parser.add_argument('--normalized', type=bool, default=False)

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataroot", type=str, default='/home/disk/disk4/celeba')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=64)
    # parser.add_argument("--pklroot", type=str, default='celeba')
    parser.add_argument('--workers', type=int, default=8)
    # parser.add_argument('-color-jittered', type=bool, default=True)
    # parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=False)
    # parser.add_argument('--normalized', type=bool, default=False)
    # parser.add_argument('--used-group', type=list, default=None)
    # parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num-concepts', type=int, default=6)
    parser.add_argument('--num-hidden', type=int, default=2)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--subsample', type=int, default=12)
    args = parser.parse_args()
    args.device = device

    # TODO: if img_size = 64, backbone network output dim = 2
    if args.img_size == 64:
        backbone_dim = 2
    elif args.imgsize == 224:
        backbone_dim = 7
    # args.device = 'cpu'
    # 3. datasets and models
    dataloaders, concept_names = celebA_dataloader.load_data(args)
    print('concept names:', concept_names)
    attr_group_dict = {concept_names[i]: [i] for i in range(len(concept_names))}
    group_size = [1] * len(concept_names)
    concept_counts = np.cumsum(np.array([0] + group_size))
    logger.info(f'concept groups: {attr_group_dict}')# , attr_group_dict)
    logger.info(f'group size: {group_size}, {concept_counts}')
    logger.info('=======================================================')
    logger.info(f'FOLDER NAME: {experiment_folder}')

    # 4. training settings
    # num_epochs = 400
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)
    if model_name == 'ViP-CBM-anchor':
        # TODO: change the backbone dims (in this case ?)
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=False, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 1, 1

    if model_name == 'ViP-CBM-anchor-NG':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=False, use_emb=False, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 1, 1

    elif model_name == 'ViP-CBM-linear':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=False, use_logits=False, anchor_model=0, shift=shift).to(device)
        alpha, beta = 1, 1

    elif model_name == 'ViP-CEM-anchor':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=True, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 1, 1

    elif model_name == 'ViP-CEM-margin':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=True, use_logits=True, anchor_model=2, shift='symmetric').to(device)
        alpha, beta = 1, 1

    elif model_name == 'ViP-CEM-anchor-NG':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=False, use_emb=True, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 1, 1

    elif model_name == 'ViP-CEM-anchor-LP':
        model = SemanticCBM(backbone_dim, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=False, use_group=True, use_emb=True, use_logits=True, anchor_model=0, shift=shift).to(device)
        alpha, beta = 1, 1

        
    elif model_name == 'jointCBM-nonlinear':
        model = CBM(backbone_dim, n_classes, n_concepts, emb_dim, 128, use_sigmoid=False).to(device)
        alpha, beta = 1, 1

    elif model_name == 'jointCBM-linear':
        model = CBM(backbone_dim, n_classes, n_concepts, emb_dim, None, use_sigmoid=False).to(device)
        alpha, beta = 1, 1

    elif model_name == 'CEM':
        model = CEM(backbone_dim, n_classes, n_concepts, emb_dim, use_sigmoid=False).to(device)
        alpha, beta = 1, 1

    elif model_name == 'ProbCBM':
        model = ProbCBM(backbone_dim, n_classes, n_concepts, emb_dim, device, use_sigmoid=False).to(device)
        alpha, beta = 1, 1

    # print(model.n_concepts, model.use_group, list(model.embeddings.parameters()))

    # load model
    model_path = os.path.join(checkpoint_dir, model_filename)
    model.load_state_dict(torch.load(model_path))

    # intervene
    test_dataloader = dataloaders['test']

    group_name = []
    concept_index = []
    group_index = []

    acc_list = []
    # test
    acc_0 = test_model(model, test_dataloader, device)
    logger.info(f'original prediction: task acc: {acc_0:.4f}')
    acc_list.append(acc_0)

    for key, value in attr_group_dict.items():
        group_name.append(key)
        concept_index = concept_index + value
        group_index.append(value)
        group_num = len(group_name)
        logger.info(f'intervene group:{group_name}, concept index ({len(concept_index)} concepts) {group_index}')
        # try:
        if 'ViP' in model_name:
            if model.use_group:
                acc = intervene(model, test_dataloader, device, group_index)
            else:
                acc = intervene(model, test_dataloader, device, concept_index)
        else: # left for baselines
            acc = intervene(model, test_dataloader, device, concept_index)
        logger.info(f'intervene group count {group_num}, task acc: {acc:.4f}')

        acc_list.append(acc)
        # except Exception as e:
        #     logger.info(f'error in intervene group count {group_num}: {e}')
        #     continue
        # acc = intervene(model, test_dataloader, device, concept_index)
        # logger.info(f'intervene group count {group_num}, task acc: {acc:.4d}')
    return acc_list, concept_counts
    # save final models
    

logger_root = './FinalLogger'
dataset_folder = 'CelebA'
logger_dir = os.path.join(logger_root, dataset_folder)
logger_name = 'vip-0224-intervene.log'
logger = get_logger_file(logger_dir, logger_name)

# for model_name in ['ViP-CEM-anchor', 'ViP-CEM-margin', 'jointCBM-nonlinear', 'CEM', 'ProbCBM']:
# # for model_name in ['ProbCBM']:
#     run_intervene(logger, 12, 32, device='cuda:3', seed=3407, model_name=model_name)
#     # run_intervene(logger, 12, 32, device='cuda:3', seed=3407, model_name='ViP-CEM-anchor')

acc_dict = {}
model_name_dict = {
    'ViP-CEM-anchor-LP': 'ViP-CBM-LP',
    'ViP-CEM-anchor': 'ViP-CBM',
    'ViP-CEM-margin': 'ViP-CBM-margin',
    'jointCBM-nonlinear': 'scalar-CBM',
    'CEM':'CEM',
    'ProbCBM':'ProbCBM'
}

for model_name in ['ViP-CEM-anchor' , 'ViP-CEM-margin', 'ViP-CEM-anchor-LP', 'jointCBM-nonlinear', 'CEM', 'ProbCBM']:
# for model_name in ['ProbCBM']:
    acc_dict[model_name_dict[model_name]], concept_counts = run_intervene(logger, 12, 32, device='cuda:3', seed=42, model_name=model_name, model_filename='model-400.pth')
    # total_group_num = len(acc_dict)

print(acc_dict)
acc_array = np.array(list(acc_dict.values()))
print(acc_array.shape)
total_group_num = acc_array.shape[1]

plt.figure()
plt.rcParams.update({
    'axes.labelsize': 'xx-large',     # 坐标轴标签字号
    'xtick.labelsize': 'large',    # X轴刻度字号
    'ytick.labelsize': 'large',    # Y轴刻度字号
    'legend.fontsize': 'large',    # 图例条目字号
    'legend.title_fontsize':'x-large', # 图例标题字号
    'figure.titlesize': 'xx-large'    # 标题字号
})
plt.subplots_adjust(left=0.15, bottom=0.15) 
plt.plot(concept_counts / 6 * 100, acc_array.T, marker='.')
plt.legend(list(acc_dict.keys()))
plt.xlabel('Intervened Concepts Ratio (%)')
plt.ylabel('Class Accuracy')
plt.title('CelebA (6 concepts, 128 classes)', fontsize='xx-large')
plt.savefig('Intervention-celebA-midpoint.pdf')

# for model_name in ['CEM', 'ProbCBM', 'ViP-CEM-margin']:
#     run_main(logger, 12, 32, 400, model_name=model_name, seed=520)

# run_main(logger, 12, 32, 400, model_name='jointCBM-nonlinear', seed=520)

# for seed in [42, 24601, 2407]:

#     for model_name in ['ViP-CEM-anchor', 'ViP-CEM-margin']:
#         try:
#             run_main(logger, 12, 32, 400, model_name=model_name, seed=seed, learning_rate=5e-3)
#         except Exception as e:
#             logger.info(f'error in {model_name} with emb_dim 12, proj_dim 32, lr 5e-3')
#             logger.info(e)

# for channel in [6, 12, 24]:
#     for emb_dim in [16, 32, 64]:
#         run_main(logger, channel, emb_dim, 400, model_name='ViP-CEM-anchor-NG', seed=3407)
        
# for seed in [42, 24601, 2407]:

#     for model_name in ['ViP-CEM-anchor-LP']:
#         # try:
#         run_main(logger, 12, 32, 400, model_name=model_name, seed=seed, learning_rate=5e-3, nonlinear=False)
        # except Exception as e:
        #     logger.info(f'error in {model_name} with emb_dim 12, proj_dim 32, lr 5e-3')
        #     logger.info(e)


# for seed in [3407, 42, 24601]:
#     for model_name in ['jointCBM-nonlinear', 'CEM', 'ProbCBM']:
#         run_main(logger, 12, 32, 400, model_name=model_name, seed=seed)

# for seed in [2407, 42, 24601, 3407]:
#     run_main(logger, 12, 32, 400, model_name='ViP-CEM-anchor-NG', seed=seed)


# #==============================================================================
# for seed in [2407, 42, 24601, 3407]:
#      # for model_name in ['ViP-CEM-anchor-LP', 'ViP-CEM-anchor-NG']:
#      try:
#          run_main(logger, 12, 32, 400, model_name='ViP-CEM-anchor-LP', seed=seed)
#      except:
#          logger.info(f'error in ViP-CBM-LP with seed {seed}')
#          continue
