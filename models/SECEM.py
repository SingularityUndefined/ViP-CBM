import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, ResNet
import math

'''
including modules for semantic embedding, selections:
    - use group / no groups
    - joint / independent / sequential
    - randint policy / no randint
    - use embeddings / no embeddings
Modules:
    CtoY_classifier: from (N, n_concepts) to (N, n_classes), a simple linear / with one hidden layer; embedding also considered
    SemanticCBM: 
'''
class CtoY_classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=None):
        super().__init__()
        if hidden_dim != None:
            self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_dim, num_classes))
        else:
            self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        y = self.fc(x)
        return y

class CtoY_embedding(nn.Module):
    def __init__(self, input_dim, num_classes, emb_dim, device, hidden_dim=None):
        super().__init__()
        if hidden_dim != None:
            self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_dim, emb_dim))
        else:
            self.fc = nn.Linear(input_dim, emb_dim)
        # TODO: max norm??
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.device = device
        self.n_classes = num_classes
        # scale
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self, x):
        g = self.emb(torch.LongTensor(range(self.n_classes)).to(self.device)) # in (n_classes, emb_dim)
        h = self.fc(x).unsqueeze(1) # in (N, 1, emb_dim)
        d = self.alpha * torch.norm(h - g, p=2, dim=-1) # in (N, n_classes)
        return -d # logits, if probs, use F.softmin
    
class LinearSep(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x, c):
        pass

class AnchorModel(nn.Module):
    def __init__(self, emb_dim,  device:torch.device, shift:bool=True, randint:float|None=None) -> None:
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(2, emb_dim), requires_grad=True)
        nn.init.trunc_normal_(self.anchors, std=1.0 / math.sqrt(emb_dim))
        self.shift = shift
        self.device = device
        if randint is not None:
            assert 0 < randint < 1, 'randint must in (0, 1)'
        self.randint = randint
        if shift:
            self.margin = nn.Parameter(0.5 * torch.ones(1), requires_grad=True)
    def forward(self, x, c):# x in (N, n_c, channels)
        N, k = x.size(0), x.size(1)
        dist_n, dist_p = torch.norm(x - self.anchors[0], p=2, dim=-1), torch.norm(x - self.anchors[1], p=2, dim=-1)
        d = dist_p - dist_n # in (N, k)
        c_sgn = (c - 0.5) * 2
        if self.shift != 'none':
            if self.margin < 0:
                self.margin = nn.Parameter(0.1 * torch.ones(1).to(self.device), requires_grad=True)
            # only when training?
            if self.training:
                if self.shift:
                    d = d + self.margin * c_sgn
                # randInt
                if self.randint is not None:
                    probs = torch.rand(N, k)
                    mask = (probs < self.randint)
                    # update intervention
                    wrong_prediction = (c_sgn[mask] * d[mask] < 0)
                    
        loss = F.relu(c_sgn * d).sum(dim=-1)
        # p = F.sigmoid(d) # in (N, k)
        # return loss and 
        return d, loss # in (N, k, 1), logits, align with linear model

class oneAnchorModel(nn.Module):
    def __init__(self, emb_dim, scale, device, epsilon=1, gamma=1) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.scale = scale
        self.device = device
        if scale:
            self.epsilon = nn.Parameter(epsilon * torch.ones(1), requires_grad=True)
        else:
            self.epsilon = epsilon
    def forward(self, x, c):# x in (N, n_c, channels)
        dist = torch.norm(x, p=2, dim=-1)
        if self.scale:
            if self.epsilon < 0: # set 1
                self.epsilon = nn.Parameter(torch.ones(1).to(self.device), requires_grad=True)
        # logits = gamma * log (epsilon * dist)
        logits = self.gamma * torch.log(self.epsilon * dist)
        return logits

# NOTICE: attr_group_dict is group dict after trim, see utils.py
'''
class semanticCBM
- including joint and other types
- including use anchor models or not
- including use group or not
- including use embedding prediction as middel layer or not 
- including use class embedding model or not
'''
class SemanticCEM(nn.Module):
    def __init__(self, 
                 backbone_dim:int, 
                 channels:int, 
                 emb_dim:int, 
                 attr_group_dict:dict, 
                 device:torch.device, 
                 model_type:str, 
                 n_classes:int | None, 
                 # randInt policy
                 randInt:float | None=None,
                 # setting args
                 use_group:bool=True, 
                 nonlinear:bool=False, 
                 use_logits:bool=False, 
                 use_emb:bool=False, 
                 # hidden args
                 class_emb_dim:int | None=None, 
                 hidden_dim:int | None=None,
                 # backbone model args
                 pretrained=True, 
                 backbone_CNN=resnet34, 
                 backbone_weight=ResNet34_Weights.DEFAULT, 
                 # anchor model, better performance
                 anchor_model:int=0, 
                 scale:bool=True, 
                 shift:str='none',
                 epsilon=1,
                 gamma=1):
        super().__init__()
        # to select used groups
        self.n_classes = n_classes
        self.class_emb_dim = class_emb_dim
        
        self.attr_group_dict = attr_group_dict
        self.group_size = [len(v) for v in self.attr_group_dict.values()]
        self.n_concepts = sum(self.group_size)

        self.device = device
        self.model_type = model_type
        assert model_type in ['joint', 'sequential', 'independent'], 'model type not in joint, sequential or independent'

        self.backbone_dim = backbone_dim
        self.channels = channels
        self.emb_dim = emb_dim
        self.nonlinear = nonlinear
        self.use_group = use_group
        
        # feature extractor
        if pretrained:
            backbone = backbone_CNN(weights=backbone_weight)
        else:
            backbone = backbone_CNN(weights=None)
        # set layers evaluates
        self.in_channels = backbone.fc.in_features
        layers = list(backbone.children())[:-2]
        self.feature_extractors = nn.Sequential(*layers)
        if pretrained:
            self.feature_extractors.eval()
        # links to our work, use a 1x1conv to align channels, a linear fc to align embedding dim
        self.conv1x1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, stride=1)
        self.align_fc = nn.Linear(backbone_dim ** 2, self.emb_dim)

        assert self.n_concepts > 0, 'no concept input'
        # embeddings, # TODO: max norm?
        if self.use_group:
            self.embeddings = nn.ModuleList([nn.Embedding(k, self.emb_dim) for k in self.group_size])
        else:
            self.embeddings = nn.Embedding(self.n_concepts, self.emb_dim)
        # nonlinear, use norm projections
        
        if self.model_type == 'joint':
            assert self.n_classes is not None, 'no classes input'
            if self.class_emb_dim is None:
                self.task_model = CtoY_classifier(self.emb_dim * self.n_concepts, self.n_classes, hidden_dim)
            else:
                self.task_model = CtoY_embedding(self.emb_dim * self.n_concepts, self.n_classes, self.class_emb_dim, self.device, hidden_dim)

    def forward(self, x, c): #args define whether to have c
        N = x.size(0) # x in (N, 3, 224, 224)
        z = self.feature_extractors(x) # in (N, 512, 7, 7)
        z = self.conv1x1(z) # in (N, channels, 7, 7)
        z = self.align_fc(torch.flatten(z, start_dim=-2, end_dim=-1)) # in (N, channels, emb_dim)
        if self.use_group:
            u_list = []
            count = 0
            for (i, k) in enumerate(self.group_size):
                c_k = c[:, count:count+k]
                v_k = self.embeddings[i](torch.LongTensor(range(k)).to(self.device)) # in (N, k, emb_dim)
                u_k = torch.matmul(z, v_k.T) # in (N, channels, k)
                # nonlinear
                if self.nonlinear:
                    u_k = F.relu(u_k + 1 / torch.norm(z, p=2, dim=-1).unsqueeze(-1) * u_k) # u_c = z_c v (1 + 1 / norm(z_c))
                u_k = u_k.transpose(-2, -1) # in (N, k, channels)
                # TODO: 
                # TODO: randint policy, loss returned similarly?
                u_list.append(u_k)
                # TODO: randint policy, loss returned similarly?
            u = torch.cat(u_list, dim=-2)

        else:
            v = self.embeddings(torch.LongTensor(range(self.n_concepts)).to(self.device)) # in (n_c, emb_dim)
            u = torch.matmul(z, v.T) # in (N, channels, n_c)
            if self.nonlinear:
                u = F.relu(u + 1 / torch.norm(z, p=2, dim=-1).unsqueeze(-1) * u) # u_c = relu(z_c v + z_c v / norm(z_c))
            u = u.transpose(-2, -1) 
        # task model     
        if self.model_type == 'joint':
            y_pred = self.task_model(u.flatten(-2))
            return u, y_pred
        else:
            return u
            
class JointLoss(nn.Module):
    def __init__(self, alpha, beta=1, use_concept_logit=False, use_triplet_loss=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_concept_logit = use_concept_logit
        self.use_triplet_loss = use_triplet_loss
    def forward(self, pred, target):
        c_pred, y_pred = pred[0], pred[1]
        c, y = target[0], target[1]
        # concept prediction loss (MLC)
        if self.use_triplet_loss:
            pass
        # TODO: triplet loss
        elif self.use_concept_logit:
            concept_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            concept_loss_fn = nn.BCELoss(reduction='mean')
        concept_loss = concept_loss_fn(c_pred, c)
        # target prediction loss
        task_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        task_loss = task_loss_fn(y_pred, y)
        
        loss = concept_loss * self.alpha + self.beta * task_loss
        return loss, concept_loss, task_loss
    


        
 # TODO: Joint loss with triple loss, full contrastive learning
 # TODO: weight revisit, with dataset creation           







        

