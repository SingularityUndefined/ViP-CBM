import torch
import torch.nn as nn
from torch import optim
from utils import *
from train.train import train
from models.SECBM import *
from models.baseline_models import *
from dataloaders import cub_dataloader

import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
import random
import argparse
# 1. logging settings

def run_main(logger, channel, emb_dim, num_epochs=400, shift='none', nonlinear=True, model_name='ViP-CBM', seed=3407, dataset_folder='CUB', learning_rate=1e-2):
    n_classes = 200
    n_concepts = 112
    seed_torch(seed)
    log_root = f'SE-CBM-group/FinalLogs_0224/Grouping'
    # logger_root = 'SE-CBM-group/FinalLogger'
    checkpoint_root = 'SE-CBM-group/FinalCheckpoints_0224'

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
    parser.add_argument("--dataroot", type=str, default='CUB')
    parser.add_argument("--batch-size", type=int, default=128)
    # resized to 224 x 224
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pklroot", type=str, default='CUB/class_attr_data_10')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('-color-jittered', type=bool, default=True)
    parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=False)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--used-group', type=list, default=None)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    # args.device = 'cpu'
    # 3. datasets and models
    dataloaders, attr2index, class2index, attr_group_dict, group_size = cub_dataloader.load_data(args)
    logger.info(f'concept groups: {attr_group_dict}')# , attr_group_dict)
    logger.info(f'group size: {group_size}')
    logger.info('=======================================================')
    logger.info(f'FOLDER NAME: {experiment_folder}')

    # 4. training settings
    # num_epochs = 400
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)
    if model_name == 'ViP-CBM-anchor':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=False, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
    if model_name == 'ViP-CBM-anchor-NG':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=False, use_emb=False, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
    elif model_name == 'ViP-CBM-linear':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=False, use_logits=False, anchor_model=0, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta, use_concept_logit=False).to(device)
    elif model_name == 'ViP-CEM-anchor':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=True, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
    elif model_name == 'ViP-CEM-margin':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=True, use_emb=True, use_logits=True, anchor_model=2, shift='symmetric').to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
    elif model_name == 'ViP-CEM-anchor-NG':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=nonlinear, use_group=False, use_emb=True, use_logits=True, anchor_model=2, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
    elif model_name == 'ViP-CEM-anchor-LP':
        model = SemanticCBM(7, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=False, use_group=True, use_emb=True, use_logits=True, anchor_model=0, shift=shift).to(device)
        alpha, beta = 5, 1
        criterion = JointLoss(alpha, beta).to(device)
        
    elif model_name == 'jointCBM-nonlinear':
        model = CBM(7, n_classes, n_concepts, emb_dim, 128, use_sigmoid=False).to(device)
        alpha, beta = 5, 1
        criterion = CBM_loss(alpha, beta, use_sigmoid=False).to(device)
    elif model_name == 'jointCBM-linear':
        model = CBM(7, n_classes, n_concepts, emb_dim, None, use_sigmoid=False).to(device)
        alpha, beta = 5, 1
        criterion = CBM_loss(alpha, beta, use_sigmoid=False).to(device)
    elif model_name == 'CEM':
        model = CEM(7, n_classes, n_concepts, emb_dim, use_sigmoid=False).to(device)
        alpha, beta = 5, 1
        criterion = CBM_loss(alpha, beta, use_sigmoid=False).to(device)
    elif model_name == 'ProbCBM':
        model = ProbCBM(7, n_classes, n_concepts, emb_dim, device, use_sigmoid=False).to(device)
        alpha, beta = 5, 1
        criterion = CBM_loss(alpha, beta, use_sigmoid=False).to(device)
    # print(model.n_concepts, model.use_group, list(model.embeddings.parameters()))
    # criterion = JointLoss(alpha, beta).to(device)
    # optimizer = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
    # multistep LR
    # optim.lr_scheduler.MultiStepLR(optimizer, [80, 160], gamma=0.2)
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of learnable parameters: {num_learnable_params}")
    print(f"Number of non-learnable parameters: {num_non_learnable_params}")

    if 'ViP' in model_name:
        train(model, 'joint', device, dataloaders, criterion, optimizer, attr_group_dict, writer, logger, num_epochs, checkpoint_dir, anchor_model=2) 
    else:
        train(model, 'joint', device, dataloaders, criterion, optimizer, attr_group_dict, writer, logger, num_epochs, checkpoint_dir, anchor_model=0) 
    
    # save final models
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model-400.pth'))

logger_root = 'SE-CBM-group/FinalLogger'
dataset_folder = 'CUB'
logger_dir = os.path.join(logger_root, dataset_folder)
logger_name = '0224.log'
logger = get_logger_file(logger_dir, logger_name)



# for model_name in ['CEM', 'ProbCBM', 'ViP-CEM-margin']:
#     run_main(logger, 12, 32, 400, model_name=model_name, seed=520)

# run_main(logger, 12, 32, 400, model_name='jointCBM-nonlinear', seed=520)
try:
    run_main(logger, 12, 32, 400, model_name='ViP-CEM-anchor-NG', seed=3407, learning_rate=1e-2)
except Exception as e:
    logger.info(f'error in ViP-CBM-anchor-NG with 12, 32, 1e-2')
    logger.info(e)
try:
    run_main(logger, 24, 32, 400, model_name='ViP-CEM-anchor-NG', seed=3407, learning_rate=1e-2)
except Exception as e:
    logger.info(f'error in ViP-CBM-anchor-NG with 24, 32, 1e-2')
    logger.info(e)
try:
    run_main(logger, 24, 32, 400, model_name='ViP-CEM-anchor-NG', seed=3407, learning_rate=5e-3)
except Exception as e:
    logger.info(f'error in ViP-CBM-anchor-NG with 24, 32, 5e-3')
    logger.info(e)
# for seed in [42, 24601, 3407]:
#     for model_name in ['ViP-CEM-anchor', 'ViP-CEM-margin']:
#         run_main(logger, 12, 32, 400, model_name=model_name, seed=seed)
#         # run_main(12, 32, 400, model_name=model_name)
#         # run_main(logger, 8, 32, 400, model_name=model_name, seed=seed)

# for seed in [42, 24601]:
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
