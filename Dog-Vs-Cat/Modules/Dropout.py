import torch
import torch.nn as nn
from torch.autograd import Variable

class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x

def dropout(p=None, dim=None, method='standard'):
    # if method == 'standard':
    #     return nn.Dropout(p)
    if method == 'gaussian':
        return GaussianDropout(p/(1-p))
    # elif method == 'variational':
    #     return VariationalDropout(p/(1-p), dim)