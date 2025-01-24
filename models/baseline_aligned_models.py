import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, ResNet

'''
basic CBM models as baselines, including CBM (hard, soft, joint), CEM, ProbCBM (if taken into consideration)?

randint policies (from CEM) included in model, which in fact only accelerates the converging speed?
'''

class CBM(nn.Module):
    def __init__(self, backbone_dim, n_classes, n_concepts, channels, emb_dim, hidden_size,pretrained=True, joint=True, use_sigmoid=True):
        super().__init__()
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            # backbone.eval()
        else:
            backbone = resnet34(weights=None)
        self.emb_dim = emb_dim
        self.channels = channels
        self.in_channels = backbone.fc.in_features
        # extractor
        layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)

        if pretrained:
            self.feature_extractor.eval()

        self.backbone_dim = backbone_dim
        self.conv1x1 = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=1, stride=1)
        self.align_fc = nn.Linear(backbone_dim ** 2, self.channels)
        # same feature nums with only linear
        if hidden_size == None:
            self.concept_predictor = nn.Linear(emb_dim * self.channels, n_concepts)
        else:
            self.concept_predictor = nn.Sequential(nn.Linear(emb_dim * self.channels, hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size, n_concepts),
                                               )
        # self.concept_predictor = nn.Linear(self.channels * backbone_dim ** 2, n_concepts)
        # TODO: sigomoid or not? CURRENTLY: no sigmoid, with Koh 2020 percentile 0.95 intervene
        self.joint = joint
        self.label_predictor = nn.Linear(n_concepts, n_classes)
        self.use_sigmoid = use_sigmoid
    
    def forward(self, x, c):
        z = self.feature_extractor(x) # in (N, 512, 2, 2)
        z = self.conv1x1(z) # in (N, 32, 2, 2)
        z = self.align_fc(torch.flatten(z, start_dim=-2, end_dim=-1))
        c = self.concept_predictor(z.flatten(start_dim=1, end_dim=-1))
        if self.use_sigmoid:
            c = F.sigmoid(c)
        y = self.label_predictor(c)
        return c, y

# sequentialCBM
class CBM_loss(nn.Module):
    def __init__(self, alpha, beta, use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_sigmoid = use_sigmoid
    def forward(self, pred, target):
        c_pred, y_pred = pred[0], pred[1]
        c, y = target[0], target[1]
        # print(c_pred.size(), y_pred.size())
        # print(c.size(), y.size())
        # concept prediction loss (MLC)
        if self.use_sigmoid:
            concept_loss_fn = nn.BCELoss(reduction='mean')
        else:
            concept_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        concept_loss = concept_loss_fn(c_pred, c)
        task_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        task_loss = task_loss_fn(y_pred, y)
        loss = concept_loss * self.alpha + task_loss * self.beta
        return loss, concept_loss, task_loss

class CEM(nn.Module):
    def __init__(self, backbone_dim, n_classes, n_concepts, emb_dim, channels, pretrained=True, joint=True, use_sigmoid=True):
        super().__init__()
        self.n_concepts = n_concepts
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            # backbone.eval()
        else:
            backbone = resnet34(weights=None)
        self.emb_dim = emb_dim
        self.channels = channels
        self.in_channels = backbone.fc.in_features
        # extractor
        layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)

        if pretrained:
            self.feature_extractor.eval()

        self.backbone_dim = backbone_dim
        self.conv1x1 = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=1, stride=1)
        self.align_fc = nn.Linear(backbone_dim ** 2, self.channels)
        # same feature 49m -> 200m * 2
        self.concept_embedding = nn.Sequential(nn.Linear(self.channels, 2 * n_concepts), nn.LeakyReLU())
        self.scoring = nn.Linear(2 * self.emb_dim, 1)
        # self.concept_predictor = nn.Linear(self.channels * backbone_dim ** 2, n_concepts)
        # TODO: sigomoid or not? CURRENTLY: no sigmoid, with Koh 2020 percentile 0.95 intervene
        self.joint = joint
        self.label_predictor = nn.Linear(n_concepts * self.emb_dim, n_classes)
        self.use_sigmoid = use_sigmoid
    def forward(self, x, c):
        z = self.feature_extractor(x) # in (N, 512, 2, 2)
        z = self.conv1x1(z) # in (N, m, 7, 7)
        z = self.align_fc(torch.flatten(z, start_dim=-2, end_dim=-1))
        c = self.concept_embedding(z).view(-1, self.emb_dim, self.n_concepts, 2).transpose(1,2) # in (N, 200, m, 2)
        logits = self.scoring(torch.flatten(c, start_dim=-2, end_dim=-1)).squeeze(-1) # (N, 200)
        p = F.sigmoid(logits)
        # print(logits.size(), p.size())
        c_hat = p.unsqueeze(2).repeat(1,1,self.emb_dim) * c[:,:,:,0] + (1 - p.unsqueeze(2).repeat(1,1,self.emb_dim)) * c[:,:,:,1] # (N, 200, m)
        # print(c_hat, c_hat.size())
        y = self.label_predictor(torch.flatten(c_hat, start_dim=-2, end_dim=-1))
        # print(y.size())
        if self.use_sigmoid:
            return p, y
        else:
            return logits, y

# ProbCBM without sampling?
class ProbCBM(nn.Module):
    def __init__(self, backbone_dim, n_classes, n_concepts, channels, emb_dim, device, pretrained=True, joint=True, use_sigmoid=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            # backbone.eval()
        else:
            backbone = resnet34(weights=None)
        self.emb_dim = emb_dim
        self.in_channels = backbone.fc.in_features
        # extractor
        layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)

        if pretrained:
            self.feature_extractor.eval()

        self.backbone_dim = backbone_dim
        self.conv1x1 = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=1, stride=1)
        self.visual_embeddings = nn.Sequential(nn.Linear(backbone_dim ** 2, channels), nn.Linear(channels, n_concepts)) # to (N, emb_dim, n_concepts)
        self.pos_embeddings = nn.Embedding(self.n_concepts, self.emb_dim)
        self.neg_embeddings = nn.Embedding(self.n_concepts, self.emb_dim)
        self.label_predictor = nn.Linear(self.n_concepts * self.emb_dim, self.n_classes)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.use_sigmoid = use_sigmoid
        self.device = device
    def forward(self, x, c):
        z = self.feature_extractor(x) # in (N, 512, 2, 2)
        z = self.conv1x1(z) # in (N, m, 7, 7)
        z = self.visual_embeddings(torch.flatten(z, start_dim=-2, end_dim=-1)).transpose(1,2) # in (N, n_concepts, emb_dim)
        c_p = self.pos_embeddings(torch.LongTensor(range(self.n_concepts)).to(self.device)) # in (N, n_concepts, emb_dim)
        c_n = self.neg_embeddings(torch.LongTensor(range(self.n_concepts)).to(self.device))

        dist_n, dist_p = torch.norm(z - c_n, p=2, dim=-1), torch.norm(z - c_p, p=2, dim=-1) # in (N, n_concepts)
        if self.alpha < 0:
            self.alpha = nn.Parameter(0.1 * torch.ones(1).to(self.device), requires_grad=True)
        logits = self.alpha * (dist_n - dist_p)
        p = F.sigmoid(logits)
        # print(c_hat, c_hat.size())
        y = self.label_predictor(torch.flatten(z, start_dim=-2, end_dim=-1))
        # print(y.size())
        if self.use_sigmoid:
            return p, y
        else:
            return logits, y