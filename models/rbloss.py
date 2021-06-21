import torch
import torch.nn as nn


class RBLoss(nn.Module):
    def __init__(self):
        super(RBLoss, self).__init__()

    def forward(self, alphas_part_max, alphas_orig):
        size = alphas_orig.shape[0]  # equals batch_size
        margin = 0.02
        loss_wt = 0.0
        for i in range(size):
            loss_wt += max(
                torch.Tensor([0]).cuda(),
                margin - (alphas_part_max[i] - alphas_orig[i]),
            )
        return loss_wt / size
