import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(self, normalized_shape=(1,), threshold=0.8, alpha=1e-1, learneable_threshold=True, learneable_alpha=True):
        """
SReLU activation (S-shaped Rectified Linear Unit), according to
arxiv.org/abs/1512.07030

The normalized_shape is similar to LayerNorm parameter (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#:~:text=Parameters-,normalized_shape,-(int%20or).
For example, if normalized_shape is (3, 5) (a 2-dimensional shape), alpha
and threshold are computed over the last 2 dimensions of the input.

---

Common use cases:

# SReLU after Linear() layer which output shape is (N, C)
activation = SReLU(normalized_shape=(C,))

# SReLU after Conv1d() layer which output shape is (N, C, L)
activation = SReLU(normalized_shape=(C, 1))

# SReLU after Conv2d() layer which output shape is (N, C, H, W)
activation = SReLU(normalized_shape=(C, 1, 1))

# SReLU after Linear() layer which output shape is (N, L1, L2, L3)
activation = SReLU(normalized_shape=(L1, L2, L3))


        :param normalized_shape: (int or iterable) Input shape from an expected
        input of size: (*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[−1]).
        If a single integer is used, it is treated as a singleton
        list, and this module will normalize over the last dimension which is
        expected to be of that specific size.
        :param threshold: (float) Initial threshold value for both sides.
        :param alpha: (float) Initial slope value for both sides.
        :param learneable_threshold: When threshold should be a learneable parameter. Default: True.
        :param learneable_alpha: When alpha should be a learneable parameter. Default: True.
        """
        super().__init__()

        # Cast to Tuple, whatever the original type
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        self.threshold_l = nn.Parameter(torch.full(normalized_shape, -threshold), requires_grad=learneable_threshold)
        self.threshold_r = nn.Parameter(torch.full(normalized_shape, +threshold), requires_grad=learneable_threshold)

        self.alpha_l = nn.Parameter(torch.full(normalized_shape, alpha), requires_grad=learneable_alpha)
        self.alpha_r = nn.Parameter(torch.full(normalized_shape, alpha), requires_grad=learneable_alpha)

    def forward(self, x):
        return torch.where(x > self.threshold_r, self.threshold_r + self.alpha_r * (x - self.threshold_r),
                           torch.where(x < self.threshold_l, self.threshold_l + self.alpha_r * (x - self.threshold_l), x))


