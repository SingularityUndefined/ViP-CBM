from torch.utils.tensorboard import SummaryWriter
import os
import logging
import numpy as np
import torch
import random


def seed_torch(seed=3407):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def trim_attr_group_dict(used_group:list, attr_group_dict:dict) -> dict:
    trim_dict = {}
    for k in used_group:
        trim_dict[k] = attr_group_dict[k]
    return trim_dict

# cauculate accuracy (concept, overall concept)
def get_sample_accs(c_out, c, group_size, n_concepts, used_sigmoid=False):
    check_num = 0
    acc_group = np.zeros((len(group_size),))
    # n_concepts = sum(group_size)
    overall_acc_group = np.zeros((len(group_size),))
    margin = 0.5 if used_sigmoid else 0
    for (i, k) in enumerate(group_size):
        acc_group[i] = ((c_out >= margin).int() == c)[:,check_num:check_num+k].sum().item() / k
        overall_acc_group[i] = (((c_out >= margin).int() != c)[:,check_num:check_num+k].sum(1) == 0).sum().item()
    
    acc = ((c_out >= margin).int() == c).sum().item() / n_concepts
    overall_acc = (((c_out >= margin).int() != c).sum(1) == 0).sum().item()
    return acc_group, overall_acc_group, acc, overall_acc

# logging

def get_logger_file(logger_dir, logger_file_name):
    os.makedirs(logger_dir, exist_ok=True)
    logger_file = os.path.join(logger_dir, logger_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_file, mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('------------logger starts---------------')
    return logger

def log_tensorboard(writer:SummaryWriter, mode:str, logger:logging, epoch, num_epochs, log_dict):
    # kwargs only accept scalars and dicts
    assert mode in ['train', 'val', 'test'], 'log mode not in train, val or test'
    # log to tensorboard and logging file
    logger_str = f'{mode}: Epoch [{epoch+1}/{num_epochs}]'
    for (key, value) in log_dict.items():
        log_name = key + '/' + mode
        if type(value) == dict:
            writer.add_scalars(log_name, value, epoch)
            data = list(value.values())
            # for group accuracy
            logger_str += f' - {key} (min, max): {min(data):.4f}, {max(data):.4f}'
        else:
            writer.add_scalar(log_name, value, epoch)
            logger_str += f' - {key}: {value:.4f}'
    # log to logger file
    logger.info(logger_str)
    

