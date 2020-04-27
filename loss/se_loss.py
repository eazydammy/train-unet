from torch.nn.modules.loss import _Loss
from torch import nn
from torch.autograd import Variable
import torch

class SELoss(_Loss):
    def __init__(self, nbr_classes):
        super(SELoss, self).__init__()
        self.nbr_classes = nbr_classes
        self.loss = nn.BCELoss()

    def forward(self, inputs, target):
        se_target = self.__get_batch_label_vector(target).type_as(inputs)
        return self.loss(torch.sigmoid(inputs), se_target)

    def __get_batch_label_vector(self, target):
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, self.nbr_classes))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=self.nbr_classes, min=0,
                               max=self.nbr_classes-1)
            vect = hist>0
            tvect[i] = vect
        return tvect