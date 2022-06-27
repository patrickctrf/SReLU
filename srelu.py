import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(self, threshold=0.8, alpha=1e-2):
        """
SReLU activation (S-shaped Rectified Linear Unit), according to
arxiv.org/abs/1512.07030
        :param threshold: Initial threshold value for both sides.
        :param alpha: Initial slope value for both sides.
        """
        super().__init__()

        self.threshold_l = nn.Parameter(torch.tensor(-threshold, requires_grad=True))
        self.threshold_r = nn.Parameter(torch.tensor(threshold, requires_grad=True))

        self.alpha_l = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        self.alpha_r = nn.Parameter(torch.tensor(alpha, requires_grad=True))

    def forward(self, x):
        return torch.where(x > self.threshold_r, self.threshold_r + self.alpha_r * (x - self.threshold_r),
                           torch.where(x < self.threshold_l,  self.threshold_l + self.alpha_r * (x - self.threshold_l), x))

