from torch.nn.modules.loss import _Loss
class SegmentationLoss(_Loss):
    def __init__(self, losses, weights):
        super(SegmentationLoss, self).__init__()

        if not isinstance(losses, tuple):
            raise RuntimeError('`losses` has to be of type tuple')
        if not isinstance(weights, tuple):
            raise RuntimeError('`weights` has to be of type tuple')
        if len(losses) != len(weights):
            raise RuntimeError('the length of `losses` and `weights` have to be equal')

        self.losses = losses
        self.weights = weights

    def forward(self, inputs, target):
        loss = 0
        if not isinstance(inputs, tuple):
            if isinstance(target, tuple):
                raise RuntimeError('If input is an object, target cannot be a tuple')
            loss = self.losses[0](inputs, target)
        else:

            for idx, pred in enumerate(inputs):
                if not isinstance(target, tuple):
                    loss += self.weights[idx] * self.losses[idx](pred, target)
                else:
                    loss += self.weights[idx] * self.losses[idx](pred, target[idx])

        return loss
