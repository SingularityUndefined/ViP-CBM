# %%
from sklearn.manifold import TSNE
import argparse
from dataloaders import awa2_dataloader
from models.SECBM import SemanticCBM
import torch
import os
from tqdm import tqdm
import numpy as np
from utils import seed_torch
# load models
## NO GROUP
# ON AWA
seed_torch(520)
os.getcwd()
os.chdir('/home/disk/qij/')
model_path = os.path.join('/home/disk/qij/SE-CBM-group/FinalCheckpoints_0714/AwA2/12_32/ViP-CEM-anchor/Seed_3407', '224.pth')

# select 3 concepts and 2 classes

# select data
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataroot", type=str, default='/home/disk/qij/AwA2/Animals_with_Attributes2')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--USE_IMAGENET_INCEPTION', type=bool, default=True)
parser.add_argument('--train-val-test-ratio', type=list, default=[0.6, 0.1, 0.3])
parser.add_argument('--used-group', type=list, default=None)
parser.add_argument('--normalized', type=bool, default=False)

parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args(args=[])

# args.device = 'cpu'
# 3. datasets and models
dataloaders, class2index, concept2index, attr_group_dict, group_size = awa2_dataloader.load_data(args)
print(f'concept groups: {attr_group_dict}')# , attr_group_dict)
print(f'group size: {group_size}')
print('=======================================================')
# dataloaders, attr2index, class2index, attr_group_dict, group_size = cub_dataloader.load_data(args)

channel = 12
emb_dim = 32
n_classes = 50
# nonlinear = False
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# %%
model = SemanticCBM(8, channel, emb_dim, attr_group_dict, device, 'joint', n_classes, nonlinear=True, use_group=True, use_emb=True, use_logits=True, anchor_model=2, shift='none', explaining=True).to(device)
model.load_state_dict(torch.load(model_path))
# train
model.eval()

unl_list = []
ulp_list = []
z_list = []
y_list = []
c_list = []
for samples in tqdm(dataloaders['test']):
    images, labels, concepts = samples['image'], samples['class_label'], samples['concept_label']
    x, y, c = images.to(device), labels.to(device).squeeze().type(torch.long), concepts.to(device)
    z, u_nl, u_lp = model(x, c)# .detach().cpu().numpy() # (N, K, m)
    c_list.append(c.detach().cpu().numpy())
    z_list.append(z.detach().cpu().numpy())
    unl_list.append(u_nl.detach().cpu().numpy())
    ulp_list.append(u_lp.detach().cpu().numpy())
    y_list.append(y.detach().cpu().numpy())

u_nl = np.concatenate(unl_list, 0)
u_lp = np.concatenate(ulp_list, 0)
z = np.concatenate(z_list, 0)
c = np.concatenate(c_list, 0)
y = np.concatenate(y_list, 0)

print(class2index, concept2index)

# %% [markdown]
# Class: douphin, cow
# 
# concepts: black, white, blue

# %%
classes = [48, 49] # cow, dophin
concepts = [0, 1, 2] # black, white, blue

# embeddings with cow and dophin
y_cow = np.argwhere(y==48)[:,0]
y_dophin = np.argwhere(y==49)[:,0]
print(y_cow.shape, y_dophin.shape)

# u_cow = u[y_cow,0:3,:]
# u_dophin = u[y_dophin,0:3,:]
z_cow = z[y_cow,0:3,:]
z_dophin = z[y_dophin,0:3,:]
print(z_cow.shape, z_dophin.shape)
n_c, n_d = z_cow.shape[0], z_dophin.shape[0]
z_cow = z_cow.reshape(n_c, -1)
z_dophin = z_dophin.reshape(n_d, -1)
z = np.concatenate([z_cow, z_dophin], 0)
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(z)
print(X_tsne.shape)
colors = ['r', 'g']
img_dir = "SE-CBM-group"
import matplotlib.pyplot as plt
plt.figure(figsize=[5,5])
plt.scatter(X_tsne[0:n_c,0], X_tsne[0:n_c,1], c='r', label='cow', alpha=0.4)
plt.scatter(X_tsne[n_c:,0], X_tsne[n_c:,1], c='g', label='douphin', alpha=0.4)
plt.legend(fontsize='x-large')
plt.savefig('SE-CBM-group/digits_tsne_Z.pdf')
# plt.show()



# %%
# define labels for each components
u_cow = u_nl[y_cow,0:3,:]
u_dophin = u_nl[y_dophin,0:3,:]
n_c, n_d = u_cow.shape[0], u_dophin.shape[0]

uc = u_cow.transpose(1,0,2).reshape(-1, 12)
ud = u_dophin.transpose(1,0,2).reshape(-1, 12)
lc = np.array([0,1,2]).repeat(n_c)
ld = np.array([3, 4, 5]).repeat(n_d)
u_emb = np.concatenate([uc, ud], 0)
l_emb = np.concatenate([lc, ld], 0)
print(u_emb, l_emb)
print(u_cow[0,0],  u_cow[1,0])

# %%
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(u_emb)
print(X_tsne.shape)
colors = ['r', 'g', 'b', 'grey', 'm', 'y']
labels = ['black', 'white', 'blue']
img_dir = "SE-CBM-group"
import matplotlib.pyplot as plt
plt.figure(figsize=[7,7])
#plt.xlim((-60,60))
#plt.ylim((-75,45))
for i in range(3):
    plt.scatter(X_tsne[i*n_c: (i+1) * n_c, 0], X_tsne[i*n_c: (i+1) * n_c, 1], c=colors[i], label="cow_" + labels[i], alpha=0.4)
for i in range(3):
    print(3 * n_c + i*n_d, 3* n_c + (i+1) * n_d)
    plt.scatter(X_tsne[3 * n_c + i*n_d: 3* n_c + (i+1) * n_d, 0], X_tsne[3 * n_c + i*n_d: 3* n_c + (i+1) * n_d, 1], c=colors[i+3], label="dophin_" + labels[i], alpha=0.4)
plt.legend(ncol=2, loc=3, fontsize='x-large')
plt.savefig('SE-CBM-group/digits_tsne_NLP.pdf')
# plt.show()


u_cow = u_lp[y_cow,0:3,:]
u_dophin = u_lp[y_dophin,0:3,:]
n_c, n_d = u_cow.shape[0], u_dophin.shape[0]

uc = u_cow.transpose(1,0,2).reshape(-1, 12)
ud = u_dophin.transpose(1,0,2).reshape(-1, 12)
lc = np.array([0,1,2]).repeat(n_c)
ld = np.array([3, 4, 5]).repeat(n_d)
u_emb = np.concatenate([uc, ud], 0)
l_emb = np.concatenate([lc, ld], 0)
print(u_emb, l_emb)
print(u_cow[0,0],  u_cow[1,0])

# %%
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(u_emb)
print(X_tsne.shape)
colors = ['r', 'g', 'b', 'grey', 'm', 'y']
labels = ['black', 'white', 'blue']
img_dir = "SE-CBM-group"
import matplotlib.pyplot as plt
plt.figure(figsize=[7,7])
#plt.xlim((-60,60))
#plt.ylim((-75,45))
for i in range(3):
    plt.scatter(X_tsne[i*n_c: (i+1) * n_c, 0], X_tsne[i*n_c: (i+1) * n_c, 1], c=colors[i], label="cow_" + labels[i], alpha=0.4)
for i in range(3):
    print(3 * n_c + i*n_d, 3* n_c + (i+1) * n_d)
    plt.scatter(X_tsne[3 * n_c + i*n_d: 3* n_c + (i+1) * n_d, 0], X_tsne[3 * n_c + i*n_d: 3* n_c + (i+1) * n_d, 1], c=colors[i+3], label="dophin_" + labels[i], alpha=0.4)
plt.legend(ncol=2, loc=3, fontsize='x-large')
plt.savefig('SE-CBM-group/digits_tsne_LP.pdf')
# plt.show()