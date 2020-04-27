from .resnet import get_resnet
from .xception import get_xception
from .vgg import get_vgg

def get_backbone(name, **kwargs):
    models = {
        'resnet50':  get_resnet(50),
        'resnet101': get_resnet(101),
        'resnet152': get_resnet(152),
        'xception': get_xception(),
        'vgg16': get_vgg(16, False),
        'vgg19': get_vgg(19, False),
        'vgg16bn': get_vgg(16, True),
        'vgg19bn': get_vgg(19, True)
    }
    return models[name](**kwargs)
