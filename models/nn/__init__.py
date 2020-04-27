from .pspnet import get_pspnet
from .encnet import get_encnet
from .deeplabv3 import get_deeplabv3
from .deeplabv3plus import get_deeplabv3plus
from .unet import get_unet
from .fcn import get_fcn

def get_model(name, kwargs):
    models = {
        'pspnet': get_pspnet,
        'encnet': get_encnet,
        'deeplabv3': get_deeplabv3,
        'deeplabv3plus': get_deeplabv3plus,
        'unet': get_unet,
        'fcn': get_fcn
    }
    return models[name](**kwargs)