import torch
import numpy as np

class SegmentationErrorMeter(object):

    def __init__(self, metrics, nbr_classes):
        self.metrics_library = {'pixAcc': (self.__batch_pixel_accuracy, self.__pixAcc),
                                'mIoU': (self.__batch_intersection_union, self.__mIoU)}
        for m in metrics:
            if m not in self.metrics_library:
                raise RuntimeError("metrics can only be choose from `pixAcc`, `mIoU` ")

        self.metrics = metrics
        self.total_correct = 0
        self.total_labeled = 0
        self.total_inter = 0
        self.total_union = 0
        self.nbr_classes = nbr_classes

    def reset(self):
        self.total_correct = 0
        self.total_labeled = 0
        self.total_inter = 0
        self.total_union = 0

    def add(self, output, target):
        # if torch.is_tensor(output):
        #     output = output.detach().cpu().squeeze().numpy()
        # if torch.is_tensor(target):
        #     target = target.detach().cpu().squeeze().numpy().astype('int64')
        for m in self.metrics:
            self.metrics_library[m][0](output, target)

    def values(self):
        return_values = []
        for m in self.metrics:
            return_values.append(self.metrics_library[m][1]())
        return tuple(return_values)

    def __batch_pixel_accuracy(self, output, target):

        _, output = torch.max(output, 1)
        output = output.cpu().numpy().astype('int64') + 1
        target = target.cpu().numpy().astype('int64') + 1

        pixel_labeled = np.sum(target > 0)
        pixel_correct = np.sum((output == target)*(target > 0))

        self.total_correct += pixel_correct
        self.total_labeled += pixel_labeled

    def __pixAcc(self):
        return 1.0 * self.total_correct / (np.spacing(1) + self.total_labeled)

    def __batch_intersection_union(self, output, target):

        _, output = torch.max(output, 1)
        # output = np.argmax(output, axis=1).astype('int64') + 1

        output = output.cpu().numpy().astype('int64') + 1
        target = target.cpu().numpy().astype('int64') + 1

        output = output * (target > 0).astype(output.dtype)

        intersection = output * (output == target)

        area_inter, _ = np.histogram(intersection, bins=self.nbr_classes, range=(1, self.nbr_classes))
        area_pred, _ = np.histogram(output, bins=self.nbr_classes, range=(1, self.nbr_classes))
        area_label, _ = np.histogram(target, bins=self.nbr_classes, range=(1, self.nbr_classes))

        area_union = area_pred + area_label - area_inter
        self.total_inter += area_inter
        self.total_union += area_union

    def __mIoU(self):
        return (1.0 * self.total_inter / (np.spacing(1) + self.total_union)).mean()

