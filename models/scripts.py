import math
import torch

c = torch.Tensor([[0,1,1,1], [1,1,0,0]])
intervene_idx = [0,1]
logits_pred = torch.Tensor([[0.2, -0.2, 0.5, 5], [-1, 3, 2,-1]])

c_k = c[:,intervene_idx]
wrong_mask = (c_k != (logits_pred[:, intervene_idx] > 0).float())
print(wrong_mask)
# print('wrong mask', wrong_mask.float().mean())
c_pred = logits_pred.clone()
print(c_pred[:, intervene_idx])
c_k_pred = c_pred[:, intervene_idx].clone()
c_k_pred[wrong_mask] = 2 * math.log(19) * c_k[wrong_mask] - math.log(19)
c_pred[:, intervene_idx] = c_k_pred

print(c_pred)