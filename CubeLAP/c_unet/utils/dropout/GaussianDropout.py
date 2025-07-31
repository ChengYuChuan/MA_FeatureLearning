import torch
from torch import nn
from torch.autograd import Variable

class GaussianDropout(nn.Module):
    """During training, randomly zeroes some of the elements of the input tensor
    with probability p using samples from a Gaussian distribution

    Args:
        p (float): Zero-out probability. Defaults to 0.5.
    """

    def __init__(self, p: float = 0.5):
        super(GaussianDropout, self).__init__()
        self.α = torch.Tensor([p / (1 - p)])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, α)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, α)
            ε = torch.randn(x.size()) * self.α + 1

            ε = Variable(ε)

            return x * ε.to(x.device)
        else:
            return x

        # #For drawing
        # if self.training and self.p > 0:
        #     alpha = torch.randn(x.size(0), x.size(1), 1, 1, 1).to(x.device) * self.p + 1
        #     return x * alpha
        # return x
