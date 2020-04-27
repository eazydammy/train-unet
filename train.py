import argparse
from models.model_manager import Manager

parser = argparse.ArgumentParser()
# model and training
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--model_pretrained', type=bool, default=False)
parser.add_argument('--model_pretrain_path', type=str, default='./logs/checkpoints/pspnet_resnet50_bk_ade20k_best.pt',
                    help='If `model_pretrained` is True, this param is necessary, or else will be neglected.')
parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
parser.add_argument('--backbone_pretrained', type=bool, default=True,
                    help='If `model_pretrained` is True, this param will be neglected.')
parser.add_argument('--backbone_pretrained_path', type=str, default='./weights/resnet50.pth',
                    help='If `model_pretrained` is True, this param will be neglected.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--mode', type=str, default='train', help='options include `train` and `pred` ')
parser.add_argument('--norm', type=str, default='bn', help='options include `bn`, `gn` or `sn`')
parser.add_argument('--S', type=str, default=16, help='used in FCN, fcn8s, fcn16s or fcn32s')

# learning rate setting
parser.add_argument('--learning_rate', type=float, default=None,
                    help='learning rate. If None, learning rate will be adjusted adaptively according to the dataset')
parser.add_argument('--lr_scheduler', type=str, default='polynomial')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--supervision', default= True)
parser.add_argument('--supervision_weight', type=float, default=0.4)
parser.add_argument('--se_loss', default= True)
parser.add_argument('--se_loss_weight', type=float, default=0.2)
parser.add_argument('--os', default=8, type=int)
parser.add_argument('--sk_conn', type=bool, default= True, help='This param only takes effect if `model` is unet and `backbone` is vgg networks')

# dataset
parser.add_argument('--dataset', type=str, default='ade20k')
parser.add_argument('--data_path', type=str, default='./data/ADEChallengeData2016')
parser.add_argument('--image_size', type=int, default=473)
parser.add_argument('--dataloader_workers', type=int, default=4)
# requires the id of background category is 0
parser.add_argument('--use_background', type=bool, default=False)

# checkpoint and saving model
parser.add_argument('--checkpoint_root_path', type=str, default='./logs/checkpoints/')
parser.add_argument('--checkpoint_prefix_name', type=str, default='trainlog')
parser.add_argument('--load_checkpoint', type=bool, default=False)
parser.add_argument('--load_checkpoint_path', type=bool, default=False)

args = parser.parse_args()
manager = Manager(args)
manager.fit()