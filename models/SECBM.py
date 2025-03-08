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
    
class AnchorModel(nn.Module):
    def __init__(self, emb_dim, scale:bool, shift:str, device:torch.device) -> None:
        assert shift in ['none', 'symmetric', 'asymmetric'], 'shift not in none, symmetric or asymmetric'
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(2, emb_dim), requires_grad=True)
        nn.init.trunc_normal_(self.anchors, std=1.0 / math.sqrt(emb_dim))
        self.scale = scale
        self.shift = shift
        self.device = device
        if scale:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        if shift != 'none':
            self.margin = nn.Parameter(0.5 * torch.ones(1), requires_grad=True)
    def forward(self, x, c):# x in (N, n_c, channels)
        dist_n, dist_p = torch.norm(x - self.anchors[0], p=2, dim=-1), torch.norm(x - self.anchors[1], p=2, dim=-1)
        d = dist_n - dist_p # in (N, k)
        if self.shift != 'none':
            if self.margin < 0:
                self.margin = nn.Parameter(0.1 * torch.ones(1).to(self.device), requires_grad=True)
            # only when training?
            if self.training:
                # print(c, self.margin, (c - 0.5) * 2)
                if self.shift == 'symmetric':
                    d = d - self.margin * ((c - 0.5) * 2)
                else: # asymmetric, only encourage positive closer
                    d = d - self.margin * c
        if self.scale:
            if self.alpha < 0:
                self.alpha = nn.Parameter(0.1 * torch.ones(1).to(self.device), requires_grad=True)
            d = self.alpha * d
        # p = F.sigmoid(d) # in (N, k)
        return d.unsqueeze(-1)#  + 1e-9 # in (N, k, 1), logits, align with linear model

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
                self.epsilon = nn.Parameter(0.2 * torch.ones(1).to(self.device), requires_grad=True)
        # logits = gamma * log (epsilon * dist)
        logits = - self.gamma * torch.log(self.epsilon * dist + 1e-8)
        rev_logits = - logits
        return rev_logits.unsqueeze(-1)

# NOTICE: attr_group_dict is group dict after trim, see utils.py
'''
class semanticCBM
- including joint and other types
- including use anchor models or not
- including use group or not
- including use embedding prediction as middel layer or not 
- including use class embedding model or not
'''
class SemanticCBM(nn.Module):
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
                 gamma=1, 
                 explaining=False):
        super().__init__()
        # to select used groups
        self.explaining = explaining
        self.n_classes = n_classes
        self.class_emb_dim = class_emb_dim
        self.randInt = randInt
        if self.randInt is not None:
            assert 0 < self.randInt < 1, 'randInt not in (0, 1)'
        self.use_emb = use_emb
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
        self.use_logits = use_logits
        assert ~self.use_emb or ~self.use_logits, 'cannot use_emb and use_logit at the same time'
        assert anchor_model in [0, 1, 2], 'anchor_model not in 0, 1, 2'
        self.anchor_model = anchor_model
        # self.backbone_CNN = backbone_CNN
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
            # for param in self.feature_extractors.parameters():
            #     # if 'conv1' in name or 'bn1' in name or 'layer1' in name or 'layer2' in name:
            #     param.requires_grad = False
                        
        # links to our work, use a 1x1conv to align channels, a linear fc to align embedding dim
        self.conv1x1 = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=1, stride=1)
        self.align_fc = nn.Linear(backbone_dim ** 2, self.channels)
        assert self.n_concepts > 0, 'no concept input'
        # embeddings, # TODO: max norm?
        if self.use_group:
            self.embeddings = nn.ModuleList([nn.Embedding(k, self.emb_dim) for k in self.group_size])
        else:
            self.embeddings = nn.Embedding(self.n_concepts, self.emb_dim)
        # nonlinear, use norm projections
        if self.anchor_model == 2:
            self.concept_prediction = AnchorModel(self.channels, scale, shift, self.device)
        elif self.anchor_model == 1:
            self.concept_prediction = oneAnchorModel(self.channels, scale, self.device, epsilon, gamma)
        else:
            self.concept_prediction = nn.Linear(self.channels, 1)
        if self.model_type == 'joint':
            assert self.n_classes is not None, 'no classes input'
            if self.class_emb_dim is None:
                if self.use_emb:
                    self.task_model = CtoY_classifier(self.channels * self.n_concepts, self.n_classes, hidden_dim)
                else:
                    self.task_model = CtoY_classifier(self.n_concepts, self.n_classes, hidden_dim)
            else:
                if self.use_emb:
                    self.task_model = CtoY_embedding(self.emb_dim * self.n_concepts, self.n_classes, self.class_emb_dim, self.device, hidden_dim)
                else:
                    self.task_model = CtoY_embedding(self.n_concepts, self.n_classes, self.class_emb_dim, self.device, hidden_dim)

    def forward(self, x, c): #args define whether to have c
        N = x.size(0) # x in (N, 3, 224, 224)
        z = self.feature_extractors(x) # in (N, 512, 7, 7)
        # print(z.size())
        z = self.conv1x1(z) # in (N, emb_dim, 7, 7)
        z = self.align_fc(torch.flatten(z, start_dim=-2, end_dim=-1)) # in (N,emb_dim, channels)
        if self.use_group:
            v_list, p_list, logits_list, prd_list, logitsrd_list, u_list = [], [], [], [], [], []
            count = 0
            for (i, k) in enumerate(self.group_size):
                c_k = c[:, count:count+k]
                v_k = self.embeddings[i](torch.LongTensor(range(k)).to(self.device)) # in (N, k, emb_dim)
                v_list.append(v_k)
                # print(z.size(), v_k.size())
                # u_k = torch.matmul(z, v_k.T) # in (N, channels, k)
                u_k = torch.matmul(v_k, z) # in (N, k, channels)
                # nonlinear
                if self.nonlinear:
                    # u_k = F.relu(u_k + 1 / torch.norm(z, p=2, dim=-1).unsqueeze(-1) * u_k) # u_c = z_c v (1 + 1 / norm(z_c))
                    u_k = F.relu(u_k + 1 / (torch.norm(z, p=2, dim=1).unsqueeze(1)) * u_k)
                # u_k = u_k.transpose(-2, -1) # in (N, k, channels)
                u_list.append(u_k)

                if self.anchor_model == 0:
                    logits = self.concept_prediction(u_k).squeeze(-1)
                else:
                    logits = self.concept_prediction(u_k, c_k).squeeze(-1) # (N, k, channels) -> (N, k)                 
                # print(logits.size())
                p = F.sigmoid(logits)
                p_list.append(p)
                logits_list.append(logits)
                # randInt policy, only used when training
                if self.model_type == 'joint' and self.randInt is not None and self.training:
                    probs = torch.rand(N, k)
                    mask = (probs < self.randInt)
                    p[mask] = c_k[mask]
                    # 0.05 percentile
                    logits[mask] = (c_k[mask] - 0.5) * 2 * torch.log(19)
                    prd_list.append(p)
                    logitsrd_list.append(logits)
                    # TODO: refine u, if wrong prediction, change u to its reflection
                    # torch.where(mask)
                count += k

            c_pred = torch.cat(p_list, dim=-1) # in (N, n_c)
            # print(u_k, c_pred.max(), c_pred.min())
            logits_pred = torch.cat(logits_list, dim=-1)
            v = torch.cat(v_list, dim=-2)
            u = torch.cat(u_list, dim=-2)
            # print(u)

            if self.randInt is not None and self.training:
                crd_pred = torch.cat(prd_list, dim=-1)
                # v_pred = crd_pred.unsqueeze(-1) * v
                # TODO: refine u as 
                # v_pred = c_pred.unsqueeze(-1) * v

        else:
            v = self.embeddings(torch.LongTensor(range(self.n_concepts)).to(self.device)) # in (n_c, emb_dim)
            u = torch.matmul(v, z) # in (N, channels, n_c)
            # print(z.size(), u.size())
            if self.nonlinear:
                # u_k = F.relu(u_k + 1 / torch.norm(z, p=2, dim=1).unsqueeze(1) * u_k)
                u = F.relu(u + 1 / (torch.norm(z, p=2, dim=1).unsqueeze(1)) * u) # u_c = relu(z_c v + z_c v / norm(z_c))
            # u = u.transpose(-2, -1) # in (N, n_c, channels)
            logits_pred = self.concept_prediction(u, c).squeeze(-1) # in (N, n_c)
            c_pred = F.sigmoid(logits_pred)
            # v_pred = c_pred.unsqueeze(-1) * v
        # print(self.concept_prediction.alpha,)
        if self.explaining:
            return z, u, torch.matmul(v, z)
        if self.model_type == 'joint':
            if self.use_emb:
                y_pred = self.task_model(u.flatten(-2))
                # return c_pred, y_predsek
            else:
                if self.use_logits:
                    y_pred = self.task_model(logits_pred)
                    # return logits_pred, y_pred
                else:
                    y_pred = self.task_model(c_pred) # logits
                    # return c_pred, y_pred
            if self.use_logits:
                return logits_pred, y_pred
            else:
                return c_pred, y_pred
        else:
            if self.use_logits:
                return (logits_pred)
            elif self.use_emb:
                return c_pred, u
            else:
                return (c_pred)
            
    def intervene(self, x, c, intervene_idx, fully_intervened=False, mid_point=True, mirror=False): #args define whether to have c
        N = x.size(0) # x in (N, 3, 224, 224)
        z = self.feature_extractors(x) # in (N, 512, 7, 7)
        # print(z.size())
        z = self.conv1x1(z) # in (N, emb_dim, 7, 7)
        z = self.align_fc(torch.flatten(z, start_dim=-2, end_dim=-1)) # in (N,emb_dim, channels)
        if self.use_group:
            v_list, p_list, logits_list, prd_list, logitsrd_list, u_list = [], [], [], [], [], []
            count = 0
            for (i, k) in enumerate(self.group_size):
                c_k = c[:, count:count+k]
                v_k = self.embeddings[i](torch.LongTensor(range(k)).to(self.device)) # in (N, k, emb_dim)
                v_list.append(v_k)
                # print(z.size(), v_k.size())
                # u_k = torch.matmul(z, v_k.T) # in (N, channels, k)
                u_k = torch.matmul(v_k, z) # in (N, k, channels)
                # nonlinear
                if self.nonlinear:
                    # u_k = F.relu(u_k + 1 / torch.norm(z, p=2, dim=-1).unsqueeze(-1) * u_k) # u_c = z_c v (1 + 1 / norm(z_c))
                    u_k = F.relu(u_k + 1 / (torch.norm(z, p=2, dim=1).unsqueeze(1)) * u_k)
                # u_k = u_k.transpose(-2, -1) # in (N, k, channels)

                ################# TODO: intervene, change u to its reflection##########
                group_idx = list(range(count, count+k))
                # print('group idx, intervene idx', group_idx, intervene_idx)
                if group_idx in intervene_idx:
                    # print(f'start intervening group {i}, group idx {group_idx}')
                    assert self.anchor_model == 2, 'only support anchor model 2'
                    # c_intervene = ck # in (B, k), binarized
                    # fully intervened, u_k in (B, k, channels)
                    if fully_intervened:
                        u_k = c_k.unsqueeze(-1) * self.concept_prediction.anchors[1] + (1 - c_k).unsqueeze(-1) * self.concept_prediction.anchors[0]
                    else:
                        logits = self.concept_prediction(u_k, c_k).squeeze(-1) # (N, k, channels) -> (N, k)
                        wrong_mask = (c_k != (logits > 0).float())
                        if mid_point:
                            u_k[wrong_mask] = self.concept_prediction.anchors[0] + self.concept_prediction.anchors[1] - u_k[wrong_mask]
                        elif mirror:
                            d_anchor = self.concept_prediction.anchors[1] - self.concept_prediction.anchors[0]
                            mid_point = (self.concept_prediction.anchors[1] + self.concept_prediction.anchors[0]) / 2
                            u_k[wrong_mask] = u_k[wrong_mask] - 2 *  torch.dot(x - mid_point, d_anchor) * d_anchor / (torch.norm(d_anchor) ** 2)
                        else:
                            u_k[wrong_mask] = c_k[wrong_mask].unsqueeze(-1) * self.concept_prediction.anchors[1] + (1 - c_k[wrong_mask]).unsqueeze(-1) * self.concept_prediction.anchors[0]

                    # print(f'end intervening group {i}, group idx {group_idx}')
                u_list.append(u_k)
                # passing through    
                if self.anchor_model == 0:
                    logits = self.concept_prediction(u_k).squeeze(-1)
                else:
                    logits = self.concept_prediction(u_k, c_k).squeeze(-1) # (N, k, channels) -> (N, k)                 
                # print(logits.size())
                p = F.sigmoid(logits)
                p_list.append(p)
                logits_list.append(logits)
                # randInt policy, only used when training
                if self.model_type == 'joint' and self.randInt is not None and self.training:
                    probs = torch.rand(N, k)
                    mask = (probs < self.randInt)
                    p[mask] = c_k[mask]
                    # 0.05 percentile
                    logits[mask] = (c_k[mask] - 0.5) * 2 * torch.log(19)
                    prd_list.append(p)
                    logitsrd_list.append(logits)
                    # TODO: refine u, if wrong prediction, change u to its reflection
                    # torch.where(mask)
                count += k

            c_pred = torch.cat(p_list, dim=-1) # in (N, n_c)
            # print(u_k, c_pred.max(), c_pred.min())
            logits_pred = torch.cat(logits_list, dim=-1)
            v = torch.cat(v_list, dim=-2)
            u = torch.cat(u_list, dim=-2)
            # print(u)

            if self.randInt is not None and self.training:
                crd_pred = torch.cat(prd_list, dim=-1)
                # v_pred = crd_pred.unsqueeze(-1) * v
                # TODO: refine u as 
                # v_pred = c_pred.unsqueeze(-1) * v

        else:
            v = self.embeddings(torch.LongTensor(range(self.n_concepts)).to(self.device)) # in (n_c, emb_dim)
            u = torch.matmul(v, z) # in (N, channels, n_c)
            # print(z.size(), u.size())
            if self.nonlinear:
                # u_k = F.relu(u_k + 1 / torch.norm(z, p=2, dim=1).unsqueeze(1) * u_k)
                u = F.relu(u + 1 / (torch.norm(z, p=2, dim=1).unsqueeze(1)) * u) # u_c = relu(z_c v + z_c v / norm(z_c))
            # u = u.transpose(-2, -1) # in (N, n_c, channels)
            # intervene
            c_k = c[:, intervene_idx]
            # u_original = u.clone()
            if fully_intervened:
                u[:, intervene_idx] = c_k.unsqueeze(-1) * self.concept_prediction.anchors[1] + (1 - c_k).unsqueeze(-1) * self.concept_prediction.anchors[0]
            else:
                logits_pred = self.concept_prediction(u, c).squeeze(-1)
                wrong_mask = (c_k != (logits_pred[:, intervene_idx] > 0).float())
                u_k = u[:, intervene_idx].clone()
                if mid_point:
                    u_k[wrong_mask] = self.concept_prediction.anchors[0] + self.concept_prediction.anchors[1] - u_k[wrong_mask]
                else:
                    u_k[wrong_mask] = c_k[wrong_mask].unsqueeze(-1) * self.concept_prediction.anchors[1] + (1 - c_k[wrong_mask]).unsqueeze(-1) * self.concept_prediction.anchors[0]
                u[:, intervene_idx] = u_k
            
            # print('u changed', torch.norm(u - u_original, p=2, dim=-1).mean())

            logits_pred = self.concept_prediction(u, c).squeeze(-1) # in (N, n_c)
            c_pred = F.sigmoid(logits_pred)
            # v_pred = c_pred.unsqueeze(-1) * v
        # print(self.concept_prediction.alpha,)
        if self.explaining:
            return z, u, torch.matmul(v, z)
        if self.model_type == 'joint':
            if self.use_emb:
                y_pred = self.task_model(u.flatten(-2))
                # return c_pred, y_predsek
            else:
                if self.use_logits:
                    y_pred = self.task_model(logits_pred)
                    # return logits_pred, y_pred
                else:
                    y_pred = self.task_model(c_pred) # logits
                    # return c_pred, y_pred
        return y_pred
        #     if self.use_logits:
        #         return logits_pred, y_pred
        #     else:
        #         return c_pred, y_pred
        # else:
        #     if self.use_logits:
        #         return (logits_pred)
        #     elif self.use_emb:
        #         return c_pred, u
        #     else:
        #         return (c_pred)
            
class JointLoss(nn.Module):
    def __init__(self, alpha, beta=1, use_concept_logit=True, use_triplet_loss=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_concept_logit = use_concept_logit
        self.use_triplet_loss = use_triplet_loss
    def forward(self, pred, target):
        c_pred, y_pred = pred[0], pred[1]
        c, y = target[0], target[1]
        # print(c_pred.min(), c_pred.max(), y_pred.min(), y_pred.max())
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







        

