import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_sample_accs, log_tensorboard
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

'''
functions in the file
train_samples, eval_samples: to train/evaluate samples from a iteration of the dataloader, return loss(es) and concept prediction (class prediction also if joint training)

run_epoch: train concept prediction, training steps in one epoch. Both joint and disjoint model are considered. After each epoch plot related scalars in tensorboard, and log each criterions in logger file (and in console).

run_task_model_epoch: running task model only, when model is sequential or independent. Notice that task model is fast to converge since these are class labels.
'''

def train_sample(x, y, c, model:nn.Module, model_type:str, loss_function:nn.Module, optimizer:optim.Optimizer):
    assert model_type in ['joint', 'independent', 'sequential'], 'model type not in joint, independent and sequential'

    optimizer.zero_grad()
    if model_type == 'joint':
        c_out, y_out = model(x, c)
        loss, concept_loss, task_loss = loss_function((c_out, y_out), (c, y))
        losses = (loss, concept_loss, task_loss)
        outs = (c_out, y_out)
    else:
        c_out = model(x, c)
        # y_out = task_model(c_out)
        loss = loss_function(c_out, c)
        losses = (loss)
        outs = (c_out)
    torch.autograd.set_detect_anomaly(True)
    # with torch.autograd.detect_anomaly():
    loss.backward()
    # loss.backward()
    optimizer.step()
    return losses, outs

def eval_sample(x, y, c, model:nn.Module, model_type:str, loss_function:nn.Module):
    assert model_type in ['joint', 'independent', 'sequential'], 'model type not in joint, independent and sequential'
    with torch.no_grad():
        if model_type == 'joint':
            c_out, y_out = model(x, c)
            loss, concept_loss, task_loss = loss_function((c_out, y_out), (c, y))
            losses = (loss, concept_loss, task_loss)
            outs = (c_out, y_out)
        else:
            c_out = model(x, c)
            # y_out = task_model(c_out)
            loss = loss_function(c_out, c)
            losses = (loss)
            # outs = (c_out)
    return losses, outs

# notice: attr_group_dict is before trim
def run_epoch(model:nn.Module, model_type:str, device:torch.device, dataloaders:dict[str,DataLoader | None], loss_function:nn.Module, optimizer:optim.Optimizer, attr_group_dict:dict, writer:SummaryWriter, logger:logging, mode:str, num_epochs:int, epoch:int, anchor_model:int=0):
    assert mode in ['train', 'val', 'test'], 'mode not in train, val or test'
    assert model_type in ['joint', 'independent', 'sequential'], 'model type not in joint, independent and sequential'
    group_size = [len(v) for v in attr_group_dict.values()]
    if mode == 'train':
        model.train()
    else:
        model.eval()
    total_sample = 0
    total_loss = 0
    total_concept_loss = 0
    total_task_loss = 0
    y_correct = 0
    c_correct = 0
    c_overall_correct = 0
    c_correct_group = np.zeros((len(group_size),))
    c_overall_correct_group = np.zeros((len(group_size),))

    # iteration over dataloader
    for samples in tqdm(dataloaders[mode]):
        # print(type(samples))
        if not isinstance(samples, dict):
            samples = {
                'image': samples[0],
                'class_label': samples[1][0],
                'concept_label': samples[1][1]
            }
        images, labels, concepts = samples['image'], samples['class_label'], samples['concept_label']
        x, y, c = images.to(device), labels.to(device).squeeze().type(torch.long), concepts.to(device)
        if mode == 'train':
            # model.train()
            losses, outs = train_sample(x, y, c, model, model_type, loss_function, optimizer)
        else:
            # model.eval()
            losses, outs = eval_sample(x, y, c, model, model_type, loss_function)

        loss = losses[0]
        c_out = outs[0]
        total_loss += loss.item()
        if model_type == 'joint':
            concept_loss, task_loss = losses[1], losses[2]
            y_out = outs[1]
            total_concept_loss += concept_loss.item()
            total_task_loss += task_loss.item()
        # group and all accuracy, overall accuracy
        acc_group, overall_acc_group, acc, overall_acc = get_sample_accs(c_out, c, group_size, n_concepts=sum(group_size), used_sigmoid=True)
        c_correct += acc
        c_overall_correct += overall_acc
        c_correct_group += acc_group
        c_overall_correct_group += overall_acc_group
        # task accurcay
        if model_type == 'joint':
            _, y_pred = torch.max(y_out, 1)
            y_correct += (y_pred == y).sum().item()

        total_sample += y.size(0)
    
    # compute accuracies
    concept_acc = 100 * c_correct / total_sample
    concept_overall_acc = 100 * c_overall_correct / total_sample
    concept_group_acc = 100 * c_correct_group / total_sample
    concept_overall_group_acc = 100 * c_overall_correct_group / total_sample

    group_acc_dict = {}
    group_overall_acc_dict = {}
    for (i,k) in enumerate(attr_group_dict.keys()):
         group_acc_dict[k] = concept_group_acc[i]
         group_overall_acc_dict[k] = concept_overall_group_acc[i]
        
    avg_loss = total_loss / len(dataloaders[mode])

    # loss_dict = {'loss':avg_loss}
    log_dict = {# 'loss':avg_loss,
            'concept_acc':concept_acc,
            'concept_overall_acc':concept_overall_acc,
            'group_acc':group_acc_dict,
            'group_overall_acc':group_overall_acc_dict
            }

    if model_type == 'joint':
        task_acc = 100 * y_correct / total_sample
        avg_concept_loss = total_concept_loss / len(dataloaders[mode])
        avg_task_loss = total_task_loss / len(dataloaders[mode])
        log_dict['task_acc'] = task_acc
        log_dict['concept_loss'] = avg_concept_loss
        log_dict['task_loss'] = avg_task_loss
        log_dict['loss'] = avg_loss
    else: # sequential, independent
        log_dict['concept_loss'] = avg_loss
        # log_dict['loss']
    
    print('anchor model', anchor_model)
    if anchor_model == 2 and mode == 'train':
        log_dict['alpha'] = model.concept_prediction.alpha.item()
        log_dict['anchor_dist'] = torch.norm(model.concept_prediction.anchors[0] - model.concept_prediction.anchors[1], p=2).item()
        if model.concept_prediction.shift != 'none':
            log_dict['margin'] = model.concept_prediction.margin.item()

    elif anchor_model == 1 and mode == 'train':
        log_dict['epsilon'] = model.concept_prediction.epsilon.item()
    
    log_tensorboard(writer, mode, logger, epoch, num_epochs, log_dict)

def run_task_model_epoch(task_model:nn.Module, model_type:str, concept_model:nn.Module | None, device:torch.device, dataloaders:dict[str,DataLoader | None], loss_function:nn.Module, optimizer:optim.Optimizer | None, writer:SummaryWriter, logger:logging, mode:str, num_epochs:int, epoch:int, binary_only=False, use_logit=False, testing_independent=False):
    assert mode in ['train', 'val', 'test'], 'mode not in train, val or test'
    assert model_type in ['independent', 'sequential'], 'model type not in independent and sequential'
    if model_type == 'sequential':
        assert concept_model is not None, 'concept model is None'
        # no training concept models
        concept_model.eval()

    if mode == 'train':
        task_model.train()
    else:
        task_model.eval()

    total_sample = 0
    total_loss = 0
    y_correct = 0
    # iteration over dataloader
    for samples in dataloaders[mode]:
        images, labels, concepts = samples['image'], samples['class_label'], samples['concept_label']
        x, y, c = images.to(device), labels.to(device).squeeze().type(torch.long), concepts.to(device)
        if model_type == 'sequential':
            with torch.no_grad():
                c = concept_model(x)
                if testing_independent:
                    if binary_only:
                        margin = 0 if use_logit else 0.5
                        c = (c > margin).float()
                    elif use_logit:
                        c = F.sigmoid(c)
        if mode == 'train':
            optimizer.zero_grad()
            y_out = task_model(c)
            loss = loss_function(y_out, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                y_out = task_model(c)
                loss = loss_function(y_out, y)

        total_loss += loss.item()
        _, y_pred = torch.max(y_out, 1)
        y_correct += (y_pred == y).sum().item()
        total_sample += y.size(0)

    task_acc = 100 * y_correct / total_sample
    avg_loss = total_loss / len(dataloaders[mode])
    log_dict = {'task_acc':task_acc,
                'task_loss':avg_loss}
    log_tensorboard(writer, mode, logger, epoch, num_epochs, log_dict)


