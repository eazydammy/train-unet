## image semantic segmentation package

A image semantic segmentation toolbox (single GPU) contains some common semantic segmentation algorithms. The code is implemented by Pytorch.

### Requires

  1. pytorch >= 1.0.0
  2. python 3.x
  
### Usage

```
 python train.py --option1 value1 --option2 value2 ...
 For the list of options, please see train.py
```

### Performance

| Algorithms    | backbone | norm |dataset | batch_size | image_size | Epoch |   pixAcc    |   mIoU    |
| -------- | -------:  | :------: |:------: | :-------:  | :------: | :-------:  | :------: | :------: |
| [PSPNet](https://github.com/hszhao/PSPNet) [1]  | resnet50 |bn |ade20k | 16 | 473 | 120 |  80.04   |   41.68  |
| PSPNet  | resnet50 |bn | ade20k | 12 | 384 | 30 |   77.1   |   38.6   |
| PSPNet  | resnet50 |bn | ade20k+bk | 12 | 384 | 30 |   72.19   |   35.3   |
| [EncNet](https://github.com/zhanghang1989/PyTorch-Encoding) [2]  | resnet50 | bn | ade20k | 16 | 480 | 120 |  79.73   |   41.11  |
| EncNet  | resnet50 | bn |ade20k | 8 | 400 | 50|   77.7   |   40.3   |
| DeeplabV3 [3]  | xception | bn |ade20k | 8 | 384 | 50|   77.6   |   39.5   |
| DeeplabV3+ [4]  | xception | bn |ade20k | 8 | 384 | 50|   77.9   |   39.8   |
| FCN32s [5]  | vgg19bn | bn |ade20k | 12 | 384 | 50|   73.0   |   31.2   |

The items with hyperlinks are the experimental results from the original paper

##### Discussion and details:

  In the original paper, authors run their experiments on the standard ADE20k(150 classes, without background). 
  But I regard the background (i.e. labeled 0 in the original mask) as a category and the output dimensionality of the PSPNet is 151 in my code.
  Therefore, the performance gap mainly comes from three aspects：
  1) I add the background class to the dataset, which may lead to category imbalance problems and increases the complexity of the model.
  2) Due to limited video memory on a single GPU, I set the batch_size to 12/8 and image_size to 384/400 instead of the parameter settings in the original paper. 
  3) In addition, the experiments in the original paper used multiple GPUs, which means a larger batch_size can be set to make Synchronization Batch Normalization layers more effective.

### TODO

- [x] PSPNet
- [x] ENCNet
- [ ] ENCNet+JPU
- [x] Deeplabv3
- [x] Deeplabv3+
- [ ] RefineNet
- [ ] FPN
- [ ] LinkNet
- [ ] SegNet
- [x] FCN
- [x] Unet
- [ ] Unet++
- [ ] DenseASPP
- [ ] ICNet
- [ ] BiSeNet
- [ ] PSANet
- [ ] DANet
- [ ] OCNet
- [ ] CCNet
- [ ] ENet
- [ ] DUNet


### References
[1] [Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1612.01105)

[2] [Zhang, Hang, et al. "Context encoding for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Context_Encoding_for_CVPR_2018_paper.pdf)

[3] [Chen, Liang Chieh , et al. "Rethinking Atrous Convolution for Semantic Image Segmentation." (2017).](https://arxiv.org/abs/1706.05587)

[4] [Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European Conference on Computer Vision (ECCV). 2018.](https://arxiv.org/abs/1802.02611)

[5] [Shelhamer, Evan, J. Long, and T. Darrell. "Fully Convolutional Networks for Semantic Segmentation." IEEE Transactions on Pattern Analysis & Machine Intelligence 39.4(2017):640-651.](http://de.arxiv.org/pdf/1411.4038)
