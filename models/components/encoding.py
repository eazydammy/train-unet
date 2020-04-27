import torch
from torch.autograd import Function
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class ScaledL2(Function):

    @staticmethod
    def forward(ctx, X, C, S):
        SL = S.view(1, 1, C.size(0)) * (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
             C.unsqueeze(0).unsqueeze(0)).pow(2).sum(3)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, grad_SL):
        X, C, S, SL = ctx.saved_variables
        tmp = (2 * grad_SL * S.view(1, 1, C.size(0))).unsqueeze(3) * \
                (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0))
        GX = tmp.sum(2)
        GC = tmp.sum(0).sum(0)
        GS = (grad_SL * (SL / S.view(1, 1, C.size(0)))).sum(0).sum(0)
        return GX, GC, GS

def scaled_l2(X, C, S):
    return ScaledL2.apply(X, C, S)

class Aggregate(Function):

    @staticmethod
    def forward(ctx, A, X, C):
        ctx.save_for_backward(A, X, C)
        E = (A.unsqueeze(3) * (X.unsqueeze(2).expand(X.size(0), X.size(1),
                                                      C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0))).sum(1)
        return E

    @staticmethod
    def backward(ctx, grad_E):
        A, X, C = ctx.saved_variables
        grad_A = (grad_E.unsqueeze(1) * (X.unsqueeze(2).expand(X.size(0), X.size(1),
                                                C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0))).sum(3)
        grad_X = torch.bmm(A, grad_E)
        grad_C = (-grad_E * A.sum(1).unsqueeze(2)).sum(0)
        return grad_A, grad_X, grad_C

def aggregate(A, X, C):
    return Aggregate.apply(A, X, C)

class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.D, -1).transpose(1, 2).contiguous()
        A = F.softmax(scaled_l2(x, self.codewords, self.scale), dim=2)
        E = aggregate(A, x, self.codewords)
        return E