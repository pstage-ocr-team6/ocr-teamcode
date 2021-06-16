import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class RotateByDistribution(nn.Module):
    def __init__(self, distribution=None):
        super(RotateByDistribution, self).__init__()
        if distribution is None:
            self.distribution = torch.distributions.normal.Normal(0, 34)
        else:
            self.distribution = distribution
            
    def forward(self, img):
        degree = self.distribution.sample().item()
        return F.rotate(img, degree)