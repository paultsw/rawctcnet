"""
Implementation of Group-Norm, as specified in
Y. Wu, K. He, "Group Normalization",
https://arxiv.org/pdf/1803.08494.pdf

In particular, GroupNorm is empirically found to perform
better for smaller batch sizes (< 16).
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class GroupNorm1d(nn.Module):
    """
    Group-Norm for 1D (sequential) input.

    Credit: implementation based on modifications to code in Figure 3 of
    the "Group Normalization" paper (referenced above).
    """
    def __init__(self, dim, group_size=2, eps=1e-5):
        """Constructor for group-norm."""
        super(GroupNorm1d, self).__init__()
        self.dim = dim
        self.group_size = group_size
        if (dim < group_size) or (dim % group_size != 0):
            raise Exception("[GroupNorm1d] dim(={0}) and group_size(={1}) are incompatible")
        self.eps = eps
        # set parameters & initialize:
        self.weight = nn.Parameter(torch.rand(1,dim,1))
        self.bias = nn.Parameter(torch.rand(1,dim,1))


    def forward(self, xs):
        """
        Args:
        xs: a sequential torch Variable input of shape (N x C x T).

        Returns:
        xs: same xs, after running group-normalization.
        """
        N, C, T = xs.size()
        xs = xs.view(N, C // self.group_size, -1)
        mu = torch.mean(xs, dim=-1, keepdim=True)
        var = torch.var(xs, dim=-1, keepdim=True)
        xs = torch.div(xs - mu, torch.sqrt(var+self.eps))
        return (xs.view(N,C,T) * self.weight + self.bias)
